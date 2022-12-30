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
from clip2mesh.utils import Utils, Pytorch3dRenderer


class VertexHeatmap:
    def __init__(self, args):

        self.utils = Utils()
        self.descriptors_dir = Path(args.descriptors_dir)
        # self.output_dir = Path(args.output_dir)
        self.iou_threshold = args.iou_threshold
        self.corr_threshold = args.corr_threshold
        self.gender = args.gender
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_renderer(args.renderer_kwargs)
        self._load_def_mesh()

    def _load_renderer(self, kwargs: DictConfig):
        self.renderer = Pytorch3dRenderer(**kwargs)

    def _load_def_mesh(self):
        verts, faces, vt, ft = self.utils.get_smplx_model()
        self.def_verts = torch.tensor(verts)[None].to(self.device)
        self.def_faces = torch.tensor(faces)[None].to(self.device)
        self.def_vt = torch.tensor(vt).to(self.device)
        self.def_ft = torch.tensor(ft).to(self.device)

    @staticmethod
    def adjust_rendered_img(img: torch.Tensor):
        img = np.clip(img.cpu().numpy()[0, ..., :3] * 255, 0, 255).astype(np.uint8)
        return img

    def __call__(self):
        descriptors_generator = list(self.descriptors_dir.iterdir())
        for descriptor in descriptors_generator:
            regular = torch.tensor(np.load(descriptor / f"{descriptor.name}.npy")).float()
            inverse = torch.tensor(np.load(descriptor / f"{descriptor.name}_inverse.npy")).float()
            if regular.dim() == 1:
                regular = regular.unsqueeze(0)
            if inverse.dim() == 1:
                inverse = inverse.unsqueeze(0)
            regular_verts, _, _, _ = self.utils.get_smplx_model(betas=regular, gender=self.gender)
            inverse_verts, _, _, _ = self.utils.get_smplx_model(betas=inverse, gender=self.gender)

            # get interpolation of color from blue to red
            distances = np.linalg.norm(regular_verts - inverse_verts, axis=1)
            sorted_indices = np.argsort(distances)

            # get color map
            cmap = plt.get_cmap("jet")
            colors = cmap(np.linspace(0, 1, len(sorted_indices)))[:, :3]

            # get vertex colors
            vertex_colors = np.zeros((len(sorted_indices), 3))
            for i, index in enumerate(sorted_indices):
                vertex_colors[index] = colors[i]
            vertex_colors = torch.tensor(vertex_colors).float()

            rend_img = self.renderer.render_mesh(
                self.def_verts,
                self.def_faces,
                self.def_vt,
                self.def_ft,
                texture_color_values=vertex_colors[None].to(self.device),
            )
            rend_img = self.adjust_rendered_img(rend_img)
            rend_img = cv2.cvtColor(rend_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(descriptor / f"vertex_heatmap.png"), rend_img)


@hydra.main(config_path="../../config", config_name="vertex_heatmap")
def main(cfg: DictConfig):
    vertex_heatmap = VertexHeatmap(cfg)
    vertex_heatmap()


if __name__ == "__main__":
    main()
