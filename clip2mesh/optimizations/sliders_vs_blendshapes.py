import cv2
import json
import torch
import hydra
import logging
import numpy as np
from torch import nn
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig
from typing import Literal, Dict, Any, List
from clip2mesh.utils import Utils, Pytorch3dRenderer


class Model(nn.Module):
    def __init__(self, num_sliders: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(1, num_sliders))

    def forward(self):
        return self.weights


class BSOptimization:
    def __init__(
        self,
        bs_dir: str,
        model_path: str,
        num_coeffs: int,
        output_dir: str,
        renderer_kwargs: Dict[str, Any],
        optimizing_feature: Literal["betas", "beta", "expression_params", "shape_params"],
        model_type: Literal["flame", "smpl", "smplx"],
        gender: Literal["male", "female", "neutral"] = "neutral",
        total_steps: int = 500,
        lr: float = 0.001,
        fps: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        display: bool = False,
        write_videos: bool = False,
    ):
        self.acumulated_3d_distance = []
        self.bs_dir = bs_dir
        self.num_coeffs = num_coeffs
        self.device = device
        self.total_steps = total_steps
        self.model_type = model_type
        self.lr = lr
        self.output_dir = Path(output_dir)
        self.write_videos = write_videos
        self.gender = gender
        self.fps = fps
        self.display = display
        self.optimizing_feature = optimizing_feature
        self.utils = Utils()
        self.img_size = renderer_kwargs["img_size"]
        self.clip2mesh_mapper, self.labels = self.utils.get_model_to_eval(model_path=model_path)

        self.model = Model(len(self.labels))
        self.model.to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self._load_logger()
        self._assertions()
        self._get_renderer(renderer_kwargs)
        self._get_3dmm_model()

        self.logger.info(f"Starting optimization for {self.model_type} {self.optimizing_feature}")
        self.logger.info(f"labels: {self.labels}")

    def _load_logger(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s: - %(message)s")
        self.logger = logging.getLogger(__name__)

    def _assertions(self):
        assert self.model_type in ["flame", "smpl", "smplx"], "Model type must be flame, smpl or smplx"
        assert self.optimizing_feature in [
            "betas",
            "beta",
            "expression_params",
            "shape_params",
        ], "Optimizing feature must be betas, beta, expression_params or shape_params"

    def _get_renderer(self, kwargs):
        self.renderer = Pytorch3dRenderer(**kwargs)

    def _get_gt(self, bs_json_path: Path) -> torch.Tensor:
        with open(bs_json_path, "r") as f:
            bs_json = json.load(f)
        return torch.tensor(bs_json[self.optimizing_feature]).to(self.device)

    def _get_sliders_values(self, labels_json_path: Path) -> torch.Tensor:
        with open(labels_json_path, "r") as f:
            labels_json = json.load(f)
        return torch.tensor(list(labels_json.values()))[..., 0, 0]

    def _get_3dmm_model(self):
        if self.model_type == "flame":
            self.three_dmm_model = self.utils.get_flame_model
        elif self.model_type == "smplx":
            self.three_dmm_model = self.utils.get_smplx_model

    def _prepare_video_writer(self, video_path: Path):
        self.logger.info(f"Writing video to {video_path}")
        self.writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.img_size[0] * 2, self.img_size[0])
        )

    def write_to_video(self, img: np.ndarray):
        self.writer.write(img)

    def close_video_writer(self):
        self.writer.release()

    @staticmethod
    def adjust_rendered_img(img: torch.Tensor) -> np.ndarray:
        img = np.clip(img.cpu().numpy()[0, ..., :3] * 255, 0, 255).astype(np.uint8)
        return img

    def get_geometric_distance(self, pred_bs: torch.Tensor, gt_bs: torch.Tensor) -> float:
        pred_model_kwargs = {self.optimizing_feature: pred_bs, "device": self.device, "gender": self.gender}
        gt_model_kwargs = {self.optimizing_feature: gt_bs, "device": self.device, "gender": self.gender}

        pred_verts, _, _, _ = self.three_dmm_model(**pred_model_kwargs)
        gt_verts, _, _, _ = self.three_dmm_model(**gt_model_kwargs)

        return torch.norm(pred_verts - gt_verts, dim=1).mean().item()

    def display_result(self, pred_bs: torch.Tensor, gt_bs: torch.Tensor, loss: float):
        pred_model_kwargs = {self.optimizing_feature: pred_bs, "device": self.device, "gender": self.gender}
        gt_model_kwargs = {self.optimizing_feature: gt_bs, "device": self.device, "gender": self.gender}

        pred_rendered_img = self.renderer.render_mesh(*self.three_dmm_model(**pred_model_kwargs))
        gt_rendered_img = self.renderer.render_mesh(*self.three_dmm_model(**gt_model_kwargs))

        pred_rendered_img = self.adjust_rendered_img(pred_rendered_img.detach())
        gt_rendered_img = self.adjust_rendered_img(gt_rendered_img)

        for img, title in zip([pred_rendered_img, gt_rendered_img], ["pred", "gt"]):
            cv2.putText(
                img,
                title,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            pred_rendered_img,
            f"loss: {loss:.4f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        concatenated_img = np.concatenate([pred_rendered_img, gt_rendered_img], axis=1)
        concatenated_img = cv2.cvtColor(concatenated_img, cv2.COLOR_BGR2RGB)

        if self.write_videos:
            self.write_to_video(concatenated_img)

        concatenated_img = cv2.resize(concatenated_img, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("optimization process", concatenated_img)
        cv2.waitKey(1)

    def optimize(self):

        max_images = 10
        images_count = 0
        for img_path in Path(self.bs_dir).rglob("*.png"):
            self.logger.info(f"Optimizing {img_path.name}")
            bs_json_path = img_path.parent / f"{img_path.stem}.json"
            labels_json_path = img_path.parent / f"{img_path.stem}_labels.json"

            if self.write_videos:
                self._prepare_video_writer(self.output_dir / f"{img_path.stem}.mp4")

            gt = self._get_gt(bs_json_path)
            sliders_values = self._get_sliders_values(labels_json_path)

            progress_bar = tqdm(range(self.total_steps), total=self.total_steps)

            for _ in progress_bar:

                self.optimizer.zero_grad()
                pred = self.clip2mesh_mapper(self.model().to(self.device))
                loss = self.loss(pred, gt)
                loss.backward()
                self.optimizer.step()

                if self.display:
                    self.display_result(pred, gt, loss.item())

                progress_bar.set_description(f"total loss: {loss.item()}")

            if self.write_videos:
                self.close_video_writer()

            geometric_dist = self.get_geometric_distance(pred, gt)
            self.acumulated_3d_distance.append(geometric_dist)
            labels_to_print = {
                label[0]: value for label, value in zip(self.labels, self.model().cpu().detach().numpy()[0])
            }
            self.logger.info(f"Finished optimizing {img_path.name}")
            self.logger.info(f"Pred sliders values {labels_to_print}")
            self.logger.info(f"acctual sliders values {sliders_values}")
            self.logger.info(f"Geometric distance: {self.acumulated_3d_distance}")
            print()

            images_count += 1
            if images_count == max_images:

                break


@hydra.main(config_path="../config", config_name="sliders_vs_blendshapes")
def main(cfg: DictConfig):
    bs_optimization = BSOptimization(**cfg)
    bs_optimization.optimize()


if __name__ == "__main__":
    main()
