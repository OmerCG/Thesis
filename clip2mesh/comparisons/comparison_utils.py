import cv2
import torch
import logging
import numpy as np
import pandas as pd
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple, Literal
from pytorch3d.structures import Meshes
from clip2mesh.utils import Image2ShapeUtils, Utils


class ComparisonUtils(Image2ShapeUtils):
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
        """Load shapy predictions from pre-processed npz file"""
        relevant_keys = ["body_pose", "betas"]
        data = np.load(npz_path, allow_pickle=True)
        return {k: torch.from_numpy(v) for k, v in data.items() if k in relevant_keys}

    @staticmethod
    def get_pixie_data(pkl_path: Path) -> torch.Tensor:
        """Load pixie predictions from pre-processed pkl file"""
        data = np.load(pkl_path, allow_pickle=True)
        return torch.tensor(data["shape"][:10])[None]

    @staticmethod
    def get_spin_data(npy_path: Path) -> Dict[str, torch.Tensor]:
        """Load spin predictions from pre-processed npy file"""
        data = np.load(npy_path, allow_pickle=True)
        return torch.tensor(data)[None]

    def _perpare_comparisons_dir(self):
        """Create a directory for the comparison results"""
        self.comparison_dirs = {k: Path(v) for k, v in self.comparison_dirs.items()}

    def get_video_structure(self, num_methods: int) -> Tuple[int, int]:
        """Get the video structure for the multiview video, based on the number of methods to compare"""
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

    def calc_distances(self, body_shapes: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate the distance between the gt and the other methods"""
        losses = {}
        for k, v in body_shapes.items():
            if k == "gt":
                continue
            losses[k] = torch.linalg.norm(body_shapes["gt"] - v, dim=1).item()
        return losses

    def get_smplx_kwargs(
        self, body_shapes: Dict[str, torch.Tensor], gender: Literal["male", "female", "neutral"]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Get the smplx kwargs for the different methods -> (vertices, faces, vt, ft)"""
        smplx_kwargs = {}
        for method, body_shape in body_shapes.items():
            smplx_kwargs[method] = self._get_smplx_attributes(body_shape, gender)
        return smplx_kwargs

    def get_meshes_from_shapes(
        self, smplx_kwargs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ) -> Dict[str, Meshes]:
        """Get the meshes from the smplx kwargs"""
        meshes = {}
        for method, args in smplx_kwargs.items():
            meshes[method] = self.renderer.get_mesh(*args)
        return meshes

    def get_rendered_images(
        self, smplx_kwargs: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], angle: float
    ) -> Dict[str, np.ndarray]:
        """Render the meshes for the different methods"""
        rendered_imgs = {}
        for method, kwargs in smplx_kwargs.items():
            kwargs.update({"rotate_mesh": {"degrees": float(angle), "axis": "y"}})
            rendered_img = self.renderer.render_mesh(**kwargs)
            rendered_imgs[method] = self.adjust_rendered_img(rendered_img)
        return rendered_imgs

    def multiview_data(
        self,
        frames_dir: Path,
        smplx_kwargs: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        video_struct: Tuple[int, int],
        raw_img: np.ndarray,
    ):
        """Create the multiview frames for the different methods"""
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
                empty_img = np.zeros_like(rendered_imgs["gt"])

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
