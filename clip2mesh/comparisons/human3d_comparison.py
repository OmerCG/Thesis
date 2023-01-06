import cv2
import json
import torch
import hydra
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from omegaconf import DictConfig
from pytorch3d.structures import Meshes
from typing import Dict, Tuple, Literal
from clip2mesh.comparisons.comparison_utils import ComparisonUtils


class Human3DComparison(ComparisonUtils):
    def __init__(self, args):
        super().__init__(args)
        self._load_gt_jsons()

    def get_gt_data(self, json_id: str, frame_idx: int) -> torch.Tensor:
        data = self.gt_jsons[json_id]
        return torch.tensor(data["betas"][frame_idx])[None]

    def _load_gt_jsons(self, file_name: str = "stats.json"):
        self.gt_jsons = {}
        files_generator = list(self.gt_dir.rglob(file_name))
        for file in files_generator:
            json_id = "_".join(file.parts[-3:-1])
            with open(file, "r") as f:
                data = json.load(f)
            self.gt_jsons[json_id] = data

    def get_body_shapes(
        self, raw_img_path: Path, gender: Literal["male", "female", "neutral"]
    ) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
        """
        Assuming the following directory structure:
        - gt_dir
        |    - person
                - video
                    -take
                        - img
        """
        img_id = raw_img_path.stem
        person_id, video_id, take_id = raw_img_path.parts[-4:-1]
        take_id = take_id.split("_")[0]

        # load raw image
        raw_img = cv2.imread(str(raw_img_path))
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        # ground truth shape
        h3d_json_id = f"{person_id}_{video_id}"
        frame_idx = int(img_id)
        gt_body_shape = self.get_gt_data(h3d_json_id, frame_idx)

        # shapy prediction
        shpay_npz_path = self.comparison_dirs["shapy"] / person_id / video_id / take_id / f"{img_id}.npz"
        shapy_data = self.get_shapy_data(shpay_npz_path)
        shapy_body_shape = torch.tensor(shapy_data["betas"])[None]
        shapy_body_pose = torch.tensor(shapy_data["body_pose"])[None]

        # pixie prediction
        pixie_pkl_path = self.comparison_dirs["pixie"] / person_id / video_id / take_id / img_id / f"{img_id}_param.pkl"
        pixie_body_shape = self.get_pixie_data(pixie_pkl_path)

        # spin prediction
        spin_npy_path = self.comparison_dirs["spin"] / person_id / video_id / take_id / f"{img_id}.npy"
        spin_body_shape = self.get_spin_data(spin_npy_path)

        # clip preprocess
        encoded_image = self.clip_preprocess(Image.fromarray(raw_img)).unsqueeze(0).to(self.device)

        # our prediction
        with torch.no_grad():
            clip_scores = self.clip_model(encoded_image, self.encoded_labels[gender])[0].float()
            # clip_scores = self.normalize_scores(clip_scores, gender)
            our_body_shape = self.model[gender](clip_scores).cpu()

        return {
            "shapy": shapy_body_shape,
            "pixie": pixie_body_shape,
            "spin": spin_body_shape,
            "ours": our_body_shape,
            "body_pose": shapy_body_pose,
            "gt": gt_body_shape,
        }, raw_img

    def __call__(self):

        for person in self.raw_imgs_dir.iterdir():
            for video in person.iterdir():
                for take in video.iterdir():
                    if "frames" not in take.name:
                        continue
                    for img in take.iterdir():
                        self.logger.info(f"Processing {img}...\n")

                        output_path = self.output_path / person.name / video.name / take.name / img.stem
                        output_path.mkdir(parents=True, exist_ok=True)

                        gender = person.stem.split("_")[0]

                        body_shapes, raw_img = self.get_body_shapes(raw_img_path=img, gender=gender)
                        body_pose: torch.Tensor = body_shapes.pop("body_pose")

                        l2_losses: Dict[str, torch.Tensor] = self.calc_distances(body_shapes)

                        smplx_args: Dict[
                            str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
                        ] = self.get_smplx_kwargs(body_shapes, gender)

                        meshes: Dict[str, Meshes] = self.get_meshes_from_shapes(smplx_args)

                        frames_dir = output_path / "frames"
                        frames_dir.mkdir(exist_ok=True)

                        num_methods = len(meshes)
                        num_blocks = num_methods + 1  # +1 for the raw image
                        video_struct = self.get_video_structure(num_blocks)
                        video_shape = (self.renderer.height * video_struct[0], self.renderer.width * video_struct[1])

                        # create video from multiview data
                        if raw_img.shape[:2] != (self.renderer.height, self.renderer.width):
                            raw_img = cv2.resize(raw_img, (self.renderer.width, self.renderer.height))

                        smplx_kwargs: Dict[str, Dict[str, np.ndarray]] = self.mesh_attributes_to_kwargs(
                            smplx_args, to_tensor=True
                        )
                        self.multiview_data(frames_dir, smplx_kwargs, video_struct, raw_img)
                        self.create_video_from_dir(frames_dir, video_shape)

                        # columns are: ["image_name", "loss", "shapy", "pixie", "spin", "ours"]
                        single_img_results = pd.DataFrame.from_dict(
                            {
                                "image_name": [f"{person.name}_{video.name}_{take.name}_{img.stem}"],
                                "loss": "l2",
                                **l2_losses,
                            }
                        )
                        self.results_df = pd.concat([self.results_df, single_img_results])

                    self.results_df.to_csv(self.output_path / "results.csv", index=False)


@hydra.main(config_path="../config", config_name="human3d_comparison")
def main(cfg: DictConfig):
    h3d_comp = Human3DComparison(cfg.comparison_kwargs)
    h3d_comp()


if __name__ == "__main__":
    main()
