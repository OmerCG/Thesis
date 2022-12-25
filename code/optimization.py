import cv2
import clip
import json
import hydra
import torch
import imageio
import numpy as np
from torch import nn
from tqdm import tqdm
from PIL import Image
from omegaconf import DictConfig
from torch.nn import functional as F
from typing import Dict, Union, Tuple
from utils import Utils, ModelsFactory, Pytorch3dRenderer


class Model(nn.Module):
    def __init__(self, params_size: Tuple[int, int] = (1, 10)):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(params_size))

    def forward(self):
        return self.weights


class CLIPLoss(nn.Module):
    def __init__(self):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")

    def forward(self, image, text):
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity


class Optimization:
    def __init__(
        self,
        model_type: str,
        optimize_features: str,
        text: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        texture: str = None,
        total_steps: int = 1000,
        lr: float = 0.001,
        output_dir: str = "./",
        fps: int = 10,
        azim: float = 0.0,
        elev: float = 0.0,
        dist: float = 0.5,
    ):
        super().__init__()
        self.total_steps = total_steps
        self.device = device
        self.model_type = model_type
        self.optimize_features = optimize_features
        self.texture = texture
        self.models_factory = ModelsFactory(model_type)
        self.clip_model, self.image_encoder = clip.load("ViT-B/32", device=device)
        self.model = Model()
        self.lr = lr
        self.renderer = Pytorch3dRenderer(tex_path=texture, azim=azim, elev=elev, dist=dist)
        self.loss_fn = CLIPLoss()
        self.text = clip.tokenize(text).to(device)
        self.output_dir = output_dir
        self.fps = fps
        self.video_recorder = self.record_video(fps=fps, output_dir=output_dir, text=text)

    @staticmethod
    def record_video(fps, output_dir, text):
        video_recorder = cv2.VideoWriter(
            f"{output_dir}/{text.replace(' ', '_')}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (512, 512)
        )
        return video_recorder

    def render_image(self, parameters):
        model_kwargs = {self.optimize_features: parameters, "device": self.device}
        verts, faces, vt, ft = self.models_factory.get_model(**model_kwargs)
        return self.renderer.render_mesh(verts=verts, faces=faces[None], vt=vt, ft=ft)

    def loss(self, parameters):
        renderer_image = self.render_image(parameters)
        loss = self.loss_fn(renderer_image[..., :3].permute(0, 3, 1, 2), self.text)
        return loss, renderer_image

    def optimize(self):
        model = Model().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        pbar = tqdm(range(self.total_steps))
        prev_loss = np.inf
        for iter_idx in pbar:
            optimizer.zero_grad()
            parameters = model()
            loss, rendered_img = self.loss(parameters)
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss: {loss.item():.4f}")
            img = rendered_img.detach().cpu().numpy()[0]
            img = cv2.resize(img, (512, 512))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("image", img)
            cv2.waitKey(1)
            img_for_vid = np.clip((img * 255), 0, 255).astype(np.uint8)
            # write iter_idx on image
            cv2.putText(
                img_for_vid,
                f"iter: {iter_idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            self.video_recorder.write(img_for_vid)

            # if abs(prev_loss - loss.item()) < 1e-7:
            #     break
            prev_loss = loss.item()

        self.video_recorder.release()
        cv2.destroyAllWindows()
        return model.weights


@hydra.main(config_path="config", config_name="optimize")
def main(cfg: DictConfig):
    optimization = Optimization(**cfg.optimization_cfg)
    model_weights = optimization.optimize()
    print(model_weights)


if __name__ == "__main__":
    main()
