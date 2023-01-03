import cv2
import torch
import hydra
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import List
from omegaconf import DictConfig
from clip2mesh.utils import Utils, Pytorch3dRenderer, ModelsFactory


class VertexHeatmap:
    def __init__(self, args):

        self.utils = Utils()
        self.descriptors_dir = Path(args.descriptors_dir)
        self.iou_threshold = args.iou_threshold
        self.corr_threshold = args.corr_threshold
        self.gender = args.gender
        self.color_threshold = args.color_threshold
        self.optimize_feature = args.optimize_feature
        self.color_map = plt.get_cmap(args.color_map)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.view_angles = range(0, 360, 45)
        self.models_factory = ModelsFactory(model_type=args.model_type)

        self.num_rows, self.num_cols = self.get_collage_shape()

        self._load_renderer(args.renderer_kwargs)
        self._load_def_mesh()
        self._initialize_df()

    def _load_renderer(self, kwargs: DictConfig):
        self.renderer = Pytorch3dRenderer(**kwargs)

    def _initialize_df(self):
        self.df = pd.DataFrame(columns=["descriptor", "effect", "sorted_indices", "total_effective_vertices"])

    def _load_def_mesh(self):
        verts, faces, vt, ft = self.models_factory.get_model()
        self.def_verts = torch.tensor(verts).to(self.device)
        if self.def_verts.dim() == 2:
            self.def_verts = self.def_verts.unsqueeze(0)
        self.def_faces = torch.tensor(faces).to(self.device)
        if self.def_faces.dim() == 2:
            self.def_faces = self.def_faces.unsqueeze(0)
        self.def_vt = torch.tensor(vt).to(self.device)
        self.def_ft = torch.tensor(ft).to(self.device)

    @staticmethod
    def adjust_rendered_img(img: torch.Tensor):
        img = np.clip(img.cpu().numpy()[0, ..., :3] * 255, 0, 255).astype(np.uint8)
        return img

    def get_collage_shape(self):
        num_rows, num_cols = self.utils.get_plot_shape(len(self.view_angles))[0]
        if num_rows > num_cols:
            return num_cols, num_rows
        return num_rows, num_cols

    def get_collage(self, images_list: List[np.ndarray]) -> np.ndarray:
        imgs_collage = [cv2.cvtColor(rend_img, cv2.COLOR_RGB2BGR) for rend_img in images_list]
        collage = np.concatenate(
            [
                np.concatenate(imgs_collage[i * self.num_cols : (i + 1) * self.num_cols], axis=1)
                for i in range(self.num_rows)
            ],
            axis=0,
        )
        return collage

    def save_color_bar_with_threshold(self, path):
        scale = np.linspace(0, 1, 11)
        fig = plt.figure(figsize=(10, 1))
        plt.imshow([scale], cmap=self.color_map, aspect=0.1, extent=[0, 1, 0, 1])
        plt.axvline(x=self.color_threshold, color="black", linewidth=1)
        plt.axis("off")
        # annotate threshold
        plt.annotate(
            f"threshold: {self.color_threshold}",
            xy=(self.color_threshold, 0.5),
            xytext=(self.color_threshold + 0.1, 0.5),
            arrowprops=dict(facecolor="black", shrink=0.05),
        )
        fig.savefig(path)

    def get_model(self, shape_vec):
        model_kwargs = {self.optimize_feature: shape_vec, "gender": self.gender}
        verts, _, _, _ = self.models_factory.get_model(**model_kwargs)
        return verts

    def __call__(self):
        descriptors_generator = list(self.descriptors_dir.iterdir())
        for descriptor in descriptors_generator:
            regular = torch.tensor(np.load(descriptor / f"{descriptor.name}.npy")).float()
            inverse = torch.tensor(np.load(descriptor / f"{descriptor.name}_inverse.npy")).float()
            if regular.dim() == 1:
                regular = regular.unsqueeze(0)
            if inverse.dim() == 1:
                inverse = inverse.unsqueeze(0)
            regular_verts = self.get_model(regular)
            inverse_verts = self.get_model(inverse)

            # get interpolation of color from blue to red
            # distances = np.linalg.norm(regular_verts - self.def_verts.cpu().squeeze().numpy(), axis=-1)
            distances = np.linalg.norm(regular_verts - inverse_verts, axis=-1)
            if distances.ndim == 2:
                distances = distances.squeeze()
            sorted_indices = np.argsort(distances)

            normalized_distances = distances / distances.max()

            vertex_colors = self.color_map(normalized_distances)[:, :3]
            total_verts_above_th = (vertex_colors > self.color_map(self.color_threshold)[:3]).all(axis=-1).sum()

            vertex_colors = torch.tensor(vertex_colors).float().to(self.device)

            if vertex_colors.dim() == 2:
                vertex_colors = vertex_colors.unsqueeze(0)

            rend_imgs = []
            for angle in self.view_angles:
                rend_img = self.renderer.render_mesh(
                    self.def_verts,
                    self.def_faces,
                    self.def_vt,
                    self.def_ft,
                    texture_color_values=vertex_colors,
                    rotate_mesh={"degrees": float(angle), "axis": "y"},
                )
                rend_img = self.adjust_rendered_img(rend_img)
                rend_img = cv2.cvtColor(rend_img, cv2.COLOR_RGB2BGR)
                rend_imgs.append(rend_img)
            collage = self.get_collage(rend_imgs)
            collage = cv2.cvtColor(collage, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(descriptor / f"vertex_heatmap.png"), collage)
            temp_df = pd.DataFrame((descriptor.name, distances.sum(), sorted_indices.tolist(), total_verts_above_th)).T
            temp_df.columns = self.df.columns
            self.df = pd.concat([self.df, temp_df])

        vertex_heatmaps_dir = self.descriptors_dir / "vertex_heatmaps"
        vertex_heatmaps_dir.mkdir(exist_ok=True)
        self.df.to_csv(vertex_heatmaps_dir / "vertex_heatmaps.csv", index=False)
        for vertex_heatmap_file in self.descriptors_dir.rglob("vertex_heatmap.png"):
            shutil.copy(vertex_heatmap_file, vertex_heatmaps_dir / (vertex_heatmap_file.parent.name + ".png"))
        self.save_color_bar_with_threshold(vertex_heatmaps_dir / "color_bar_w_threshold.png")


@hydra.main(config_path="../../config", config_name="vertex_heatmap")
def main(cfg: DictConfig):
    vertex_heatmap = VertexHeatmap(cfg)
    vertex_heatmap()


if __name__ == "__main__":
    main()
