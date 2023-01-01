import cv2
import h5py
import torch
import hydra
import logging
import numpy as np
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from omegaconf import DictConfig
from pytorch3d.structures import Meshes
from typing import Dict, Tuple, Literal
from clip2mesh.utils import Utils, Image2ShapeUtils


class NeuralBodyComparison(Image2ShapeUtils):
    def __init__(self, args):
        super().__init__()
        self.raw_imgs_dir: Path = Path(args.raw_imgs_dir)
        self.comparison_dirs: Dict[str, str] = args.comparison_dirs
        self.gt_dir: Path = Path(args.gt_dir)
        self.output_path: Path = Path(args.output_path)
        self.utils: Utils = Utils(comparison_mode=True)
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        self._load_renderer(args.renderer_kwargs)
        self._load_smplx_models(**args.smplx_models_paths)
        self._load_weights(args.labels_weights)
        self._load_clip_model()
        self._encode_labels()
        self._perpare_comparisons_dir()
        self._load_results_df()
        self._load_logger()

    def _load_results_df(self):
        self.results_df = pd.DataFrame(columns=["image_name", "loss", "shapy", "pixie", "spin", "ours"])

    def _load_logger(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s: - %(message)s")
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def get_shapy_data(npz_path: Path) -> Dict[str, torch.Tensor]:
        relevant_keys = ["body_pose", "betas"]
        data = np.load(npz_path, allow_pickle=True)
        return {k: torch.from_numpy(v) for k, v in data.items() if k in relevant_keys}

    @staticmethod
    def get_pixie_data(pkl_path: Path) -> torch.Tensor:
        data = np.load(pkl_path, allow_pickle=True)
        return torch.tensor(data["shape"][:10])[None]

    @staticmethod
    def get_spin_data(npy_path: Path) -> Dict[str, torch.Tensor]:
        data = np.load(npy_path, allow_pickle=True)
        return torch.tensor(data)[None]

    @staticmethod
    def get_gt_data(h5_path: Path) -> torch.Tensor:
        data = h5py.File(h5_path, "r")
        return torch.tensor(data["betas"])[None]

    def _perpare_comparisons_dir(self):
        self.comparison_dirs = {k: Path(v) for k, v in self.comparison_dirs.items()}

    def get_body_shapes(
        self, raw_img_path: Path, gender: Literal["male", "female", "neutral"]
    ) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
        img_id = raw_img_path.stem

        # load raw image
        raw_img = cv2.imread(str(raw_img_path))
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        # ground truth shape
        nb_h5_path = self.gt_dir / img_id.split("_")[0] / "reconstructed_poses.hdf5"
        gt_body_shape = self.get_gt_data(nb_h5_path)

        # shapy prediction
        shpay_npz_path = self.comparison_dirs["shapy"] / f"{img_id}.npz"
        shapy_data = self.get_shapy_data(shpay_npz_path)
        shapy_body_shape = torch.tensor(shapy_data["betas"])[None]  # TODO: maybe convert to tensor or add dimension
        shapy_body_pose = torch.tensor(shapy_data["body_pose"])[None]

        # pixie prediction
        pixie_pkl_path = self.comparison_dirs["pixie"] / img_id / f"{img_id}_param.pkl"
        pixie_body_shape = self.get_pixie_data(pixie_pkl_path)

        # spin prediction
        spin_npy_path = self.comparison_dirs["spin"] / f"{img_id}.npy"
        spin_body_shape = self.get_spin_data(spin_npy_path)

        # clip preprocess
        encoded_image = self.clip_preprocess(Image.fromarray(raw_img)).unsqueeze(0).to(self.device)

        # our prediction
        with torch.no_grad():
            clip_scores = self.clip_model(encoded_image, self.encoded_labels[gender])[0]
            clip_scores = self.normalize_scores(clip_scores, gender)
            our_body_shape = self.model[gender](clip_scores).cpu()

        return {
            "shapy": shapy_body_shape,
            "pixie": pixie_body_shape,
            "spin": spin_body_shape,
            "ours": our_body_shape,
            "body_pose": shapy_body_pose,
            "gt": gt_body_shape,
        }, raw_img

    def calc_losses(self, body_shapes: Dict[str, torch.Tensor]) -> Dict[str, float]:
        losses = {}
        for k, v in body_shapes.items():
            if k == "gt":
                continue
            losses[k] = F.mse_loss(body_shapes["gt"], v).item()
        return losses

    def get_smplx_kwargs(
        self, body_shapes: Dict[str, torch.Tensor], gender: Literal["male", "female", "neutral"]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        smplx_kwargs = {}
        for method, body_shape in body_shapes.items():
            smplx_kwargs[method] = self._get_smplx_attributes(body_shape, "neutral")
        return smplx_kwargs

    def get_meshes_from_shapes(
        self, smplx_kwargs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ) -> Dict[str, Meshes]:
        meshes = {}
        for method, args in smplx_kwargs.items():
            meshes[method] = self.renderer.get_mesh(*args)
        return meshes

    def get_rendered_images(
        self, smplx_kwargs: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], angle: float
    ) -> Dict[str, np.ndarray]:
        rendered_imgs = {}
        for method, kwargs in smplx_kwargs.items():
            kwargs.update({"rotate_mesh": {"degrees": float(angle), "axis": "y"}})
            rendered_img = self.renderer.render_mesh(**kwargs)
            rendered_imgs[method] = self.adjust_rendered_img(rendered_img)
        return rendered_imgs

    def get_video_structure(self, num_methods: int) -> Tuple[int, int]:
        suggested_video_struct, num_imgs = self.utils.get_plot_shape(num_methods)
        if num_imgs < num_methods:
            suggested_video_struct = list(suggested_video_struct)
            while num_imgs < num_methods:
                suggested_video_struct[0] += 1
                num_imgs += 1
        video_struct = tuple(suggested_video_struct)
        if video_struct[0] > video_struct[1]:
            video_struct = (video_struct[1], video_struct[0])
        return video_struct

    def multiview_data(
        self,
        frames_dir: Path,
        smplx_kwargs: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        video_struct: Tuple[int, int],
        raw_img: np.ndarray,
    ):

        for frame_idx, angle in enumerate(range(0, 365, 5)):

            rendered_imgs: Dict[str, np.ndarray] = self.get_rendered_images(smplx_kwargs, angle)

            # add description to the image of its type (gt, shapy, pixie, spin, our)
            for method, img in rendered_imgs.items():
                cv2.putText(
                    img,
                    method,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            num_rows, num_cols = video_struct

            # if there are less methods than the video structure allows, add empty image
            # +1 because we have also the raw image
            if len(rendered_imgs) + 1 < num_rows * num_cols:
                empty_img = np.zeros_like(gt_img)
                for _ in range(num_rows * num_cols - len(rendered_imgs)):
                    rendered_imgs["empty"] = empty_img

            gt_img = rendered_imgs.pop("gt")

            row_imgs = []
            root_imgs = [raw_img, gt_img]
            cols_counter = 0
            offset = 2
            for row_idx in range(num_rows):
                if row_idx == 0:
                    row_imgs.append(cv2.hconcat(root_imgs + list(rendered_imgs.values())[: num_cols - offset]))
                else:
                    row_imgs.append(
                        cv2.hconcat(list(rendered_imgs.values())[num_cols - offset : num_cols + cols_counter])
                    )
                cols_counter += num_cols
            final_img = cv2.vconcat(row_imgs)

            final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frames_dir / f"{frame_idx}.png"), final_img)

    def __call__(self):

        for raw_img_path in self.raw_imgs_dir.iterdir():

            self.logger.info(f"Processing {raw_img_path.name}...\n")

            output_path = self.output_path / raw_img_path.stem
            output_path.mkdir(exist_ok=True)

            gender = raw_img_path.stem.split("-")[0]

            body_shapes, raw_img = self.get_body_shapes(raw_img_path, gender)
            body_pose: torch.Tensor = body_shapes.pop("body_pose")

            l2_losses: Dict[str, torch.Tensor] = self.calc_losses(body_shapes)

            # if not (output_path / "out_vid.mp4").exists():
            #     self.logger.info(f"Video for {raw_img_path.name} already exists. Skipping...\n")

            smplx_args: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = self.get_smplx_kwargs(
                body_shapes, gender
            )

            meshes: Dict[str, Meshes] = self.get_meshes_from_shapes(smplx_args)
            # TODO: add chamfer loss with and without body pose

            frames_dir = output_path / "frames"
            frames_dir.mkdir(exist_ok=True)

            num_methods = len(meshes)
            num_blocks = num_methods + 1  # +1 because we have also the raw image
            video_struct = self.get_video_structure(num_blocks)
            video_shape = (self.renderer.height * video_struct[0], self.renderer.width * video_struct[1])

            # create video from multiview data
            if raw_img.shape[:2] != (self.renderer.height, self.renderer.width):
                raw_img = cv2.resize(raw_img, (self.renderer.width, self.renderer.height))

            smplx_kwargs: Dict[str, Dict[str, np.ndarray]] = self.mesh_attributes_to_kwargs(smplx_args, to_tensor=True)
            self.multiview_data(frames_dir, smplx_kwargs, video_struct, raw_img)
            self.create_video_from_dir(frames_dir, video_shape)

            # columns are: ["image_name", "loss", "shapy", "pixie", "spin", "ours"]
            single_img_results = pd.DataFrame.from_dict({"image_name": [raw_img_path.stem], "loss": "l2", **l2_losses})
            self.results_df = pd.concat([self.results_df, single_img_results])

        self.results_df.to_csv(self.output_path / "results.csv", index=False)


@hydra.main(config_path="../config", config_name="neuralbody_comparison")
def main(cfg: DictConfig) -> None:
    neuralbody_comparison = NeuralBodyComparison(cfg.comparison_kwargs)
    neuralbody_comparison()


if __name__ == "__main__":
    main()
