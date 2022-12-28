import cv2
import json
import torch
import hydra
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import Literal
from omegaconf import DictConfig
from clip2mesh.utils import Utils, ModelsFactory


class DescriptorsAnalysis:
    def __init__(self, args):

        self.utils = Utils()
        self.top_k = args.top_k
        self.parameter = args.parameter
        self.output_dir = Path(args.output_dir)
        self.iou_threshold = args.iou_threshold
        self.corr_threshold = args.corr_threshold
        self.models_factory = ModelsFactory(args.model_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.color_1 = torch.tensor([0.7, 0.0, 0.0], device=self.device)
        self.color_2 = torch.tensor([0.0, 0.7, 0.0], device=self.device)
        self.color_combined = torch.tensor([0.7, 0.7, 0.0], device=self.device)

        self._load_df(args.jsons_dir)
        self._load_model(args.model_path)
        self._load_renderer(args.renderer_kwargs)
        self._load_model_kwargs(args.gender)

    def _load_renderer(self, kwargs):
        self.renderer = self.models_factory.get_renderer(py3d=True, **kwargs)

    def _load_model(self, model_path):
        self.model, self.labels = self.utils.get_model_to_eval(model_path)

    def _load_model_kwargs(self, gender: Literal["male", "female", "neutral"]):
        verts, faces, _, _ = self.models_factory.get_model(gender=gender)
        if isinstance(verts, np.ndarray):
            verts = torch.tensor(verts)
            faces = torch.tensor(faces)
        self.verts = verts.to(self.device)
        self.faces = faces.to(self.device)
        self.verts_color = (torch.ones(self.verts.shape[-2], 3) * 0.7)[None].to(self.device)

    def _load_df(self, jsons_dir):
        json_files = [f for f in Path(jsons_dir).rglob("*_labels.json")]
        df = pd.DataFrame()
        for json_file in tqdm(json_files, desc="Loading jsons", total=len(json_files)):
            with open(json_file, "r") as f:
                data = pd.DataFrame.from_dict(json.load(f))
            df = pd.concat([df, data])

        self.df = df.apply(lambda x: [y[0] for y in x])
        self.corr_df = self.calc_correlations()

    def calc_correlations(self) -> pd.DataFrame:
        """Calculate the correlation between each descriptor."""
        corr = self.df.corr()
        corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool_))
        corr = corr.stack().reset_index()
        corr.columns = ["label1", "label2", "correlation"]
        corr = corr[corr["correlation"].abs() > self.corr_threshold]
        corr = corr.sort_values(by="correlation", ascending=False)

        return corr

    def get_most_effective_vertices(self, coeff_idx: int) -> torch.Tensor:
        """get the most effective vertices for a specific coefficient."""
        coeff_1 = torch.ones(1, len(self.labels)) * 20.0
        coeff_2 = torch.ones(1, len(self.labels)) * 20.0

        coeff_1[0, coeff_idx] = 0.0
        coeff_2[0, coeff_idx] = 50.0

        with torch.no_grad():
            coeff_pred_1 = self.model(coeff_1.to(self.device)).cpu()
            coeff_pred_2 = self.model(coeff_2.to(self.device)).cpu()

        verts_1, _, _, _ = self.models_factory.get_model(**{self.parameter: coeff_pred_1})
        verts_2, _, _, _ = self.models_factory.get_model(**{self.parameter: coeff_pred_2})

        if isinstance(verts_1, np.ndarray):
            verts_1 = torch.tensor(verts_1)
            verts_2 = torch.tensor(verts_2)

        diff = torch.abs(verts_1 - verts_2)
        indices = diff.sum(dim=-1).topk(self.top_k).indices

        if diff.size().__len__() == 3:
            diff = diff.squeeze()
        if indices.size().__len__() == 2:
            indices = indices.squeeze()

        return indices, diff

    def save_image(self, image: np.ndarray, label_1: str, label_2: str, chosen: str):
        """Save the image to the output directory."""
        cv2.putText(
            image,
            f"Chosen: {chosen}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label_1,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label_2,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(self.output_dir / f"{label_1}_{label_2}.png"), image)

    def get_rendered_image(
        self, indices_1: torch.Tensor, indices_2: torch.Tensor, intersection: np.ndarray
    ) -> np.ndarray:
        """Get the rendered image of the vertices."""

        verts_color = self.verts_color.clone()
        verts_color[0, indices_1] = self.color_1
        verts_color[0, indices_2] = self.color_2
        verts_color[0, intersection] = self.color_combined
        image = self.renderer.render_mesh(verts=self.verts, faces=self.faces, texture_color_values=verts_color)
        image = image.detach().cpu().numpy().squeeze()
        image = np.clip(image[..., :3] * 255, 0, 255).astype(np.uint8)
        return image

    def calc_iou_vertices(self, indices_1: torch.Tensor, indices_2: torch.Tensor) -> float:
        """Calculate the intersection over union between the vertices."""
        indices_1 = indices_1.numpy().reshape(-1, 1)
        indices_2 = indices_2.numpy().reshape(-1, 1)
        intersection = np.intersect1d(indices_1, indices_2)
        union = np.union1d(indices_1, indices_2)
        iou = intersection.shape[0] / union.shape[0]
        diff_1 = np.setdiff1d(indices_1, indices_2)
        diff_2 = np.setdiff1d(indices_2, indices_1)

        return iou, diff_1, diff_2, intersection

    def __call__(self):
        out_df = pd.DataFrame(columns=["label1", "label2", "iou", "chosen"])
        for _, value in tqdm(self.corr_df.iterrows(), total=len(self.corr_df), desc="creating heatmap images"):

            chosen = None
            label_1 = value["label1"]
            label_2 = value["label2"]
            coeff_idx_1 = self.labels.index([label_1])
            coeff_idx_2 = self.labels.index([label_2])

            indices_1, all_1 = self.get_most_effective_vertices(coeff_idx_1)
            indices_2, all_2 = self.get_most_effective_vertices(coeff_idx_2)

            iou, diff_1, diff_2, inter = self.calc_iou_vertices(indices_1, indices_2)

            heatmap_image = self.get_rendered_image(diff_1, diff_2, inter)

            if iou > self.iou_threshold:
                if all_1[indices_1].sum() > all_2[indices_2].sum():
                    chosen = label_1
                else:
                    chosen = label_2
            else:
                chosen = f"{label_1} and {label_2}"
            self.save_image(heatmap_image, label_1, label_2, chosen)
            out_df = pd.concat(
                [
                    out_df,
                    pd.DataFrame.from_dict(
                        {"label1": label_1, "label2": label_2, "iou": iou, "chosen": chosen}, "index"
                    ).T,
                ]
            )

        out_df.to_csv(self.output_dir / "heatmap.csv", index=False)


@hydra.main(config_path="../../config", config_name="vertex_heatmap")
def main(cfg: DictConfig):
    vertex_heatmap = DescriptorsAnalysis(cfg)
    vertex_heatmap()


if __name__ == "__main__":
    main()
