import cv2
import clip
import hydra
import torch
import logging
import numpy as np
from torch import nn
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig
from typing import Tuple, Literal, List
from clip2mesh.utils import ModelsFactory, Pytorch3dRenderer, Utils


class Model(nn.Module):
    def __init__(self, params_size: Tuple[int, int] = (1, 10)):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(params_size))

    def forward(self):
        return self.weights


class CLIPLoss(nn.Module):
    def __init__(self, inverse: bool = False):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.inverse = inverse

    def forward(self, image, text):
        if self.inverse:
            similarity = self.model(image, text)[0] / 100
        else:
            similarity = 1 - self.model(image, text)[0] / 100
        return similarity


class Optimization:
    def __init__(
        self,
        model_type: str,
        optimize_features: str,
        text: List[str],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        texture: str = None,
        total_steps: int = 1000,
        lr: float = 0.001,
        output_dir: str = "./",
        fps: int = 10,
        azim: float = 0.0,
        elev: float = 0.0,
        dist: float = 0.5,
        gender: Literal["male", "female", "neutral"] = "neutral",
        img_out_size: Tuple[int, int] = (512, 512),
        display: bool = False,
    ):
        super().__init__()
        self.total_steps = total_steps
        self.device = device
        self.gender = gender
        self.model_type = model_type
        self.optimize_features = optimize_features
        self.texture = texture
        self.models_factory = ModelsFactory(model_type)
        self.clip_model, self.image_encoder = clip.load("ViT-B/32", device=device)
        self.model = Model()
        self.lr = lr
        self.utils = Utils()
        self.clip_renderer = Pytorch3dRenderer(tex_path=texture, azim=azim, elev=elev, dist=dist)
        self.text = text
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.display = display
        self.img_out_size = img_out_size
        self.view_angles = range(0, 360, 45)
        self.num_rows, self.num_cols = self.get_collage_shape()

        self._load_logger()

    def record_video(self, fps, output_dir, text) -> cv2.VideoWriter:
        collage_size = (self.img_out_size[0] * self.num_cols, self.img_out_size[1] * self.num_rows)
        video_recorder = cv2.VideoWriter(
            f"{output_dir}/{text.replace(' ', '_')}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, collage_size
        )
        return video_recorder

    def _load_logger(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s: - %(message)s")
        self.logger = logging.getLogger(__name__)

    def render_image(self, parameters: torch.Tensor, angle: float) -> torch.Tensor:
        model_kwargs = {self.optimize_features: parameters, "device": self.device}
        verts, faces, vt, ft = self.models_factory.get_model(gender=self.gender, **model_kwargs)
        rotate_meah_kwargs = {"degrees": float(angle), "axis": "y"}
        image_for_clip = self.clip_renderer.render_mesh(
            verts=verts, faces=faces[None], vt=vt, ft=ft, rotate_mesh=rotate_meah_kwargs
        )
        return image_for_clip

    def loss(
        self, parameters, loss_fn: CLIPLoss, text: torch.Tensor, angle: float = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        clip_renderer_image = self.render_image(parameters, angle=angle)
        loss = loss_fn(clip_renderer_image[..., :3].permute(0, 3, 1, 2), text)
        return loss, clip_renderer_image

    def get_collage_shape(self):
        num_rows, num_cols = self.utils.get_plot_shape(len(self.view_angles))[0]
        if num_rows > num_cols:
            return num_cols, num_rows
        return num_rows, num_cols

    def get_collage(self, images_list: List[np.ndarray]) -> np.ndarray:
        imgs_collage = [
            cv2.cvtColor(
                cv2.resize(rend_img.detach().cpu().numpy()[0], self.img_out_size),
                cv2.COLOR_RGB2BGR,
            )
            for rend_img in images_list
        ]
        collage = np.concatenate(
            [
                np.concatenate(imgs_collage[i * self.num_cols : (i + 1) * self.num_cols], axis=1)
                for i in range(self.num_rows)
            ],
            axis=0,
        )
        return collage

    def optimize(self):

        for word_desciptor in self.text:

            self.logger.info(f"Optimizing for {word_desciptor}...")
            output_dir = self.output_dir / word_desciptor
            output_dir.mkdir(parents=True, exist_ok=True)

            encoded_text = clip.tokenize([word_desciptor]).to(self.device)

            for phase in ["regular", "inverse"]:

                file_name = f"{word_desciptor}_{phase}.npy" if phase == "inverse" else f"{word_desciptor}.npy"
                out_file_path = output_dir / file_name
                if out_file_path.exists():
                    self.logger.info(f"File {out_file_path} already exists. Skipping...")
                    continue

                self.logger.info(f"Phase: {phase}")
                if phase == "inverse":
                    loss_fn = CLIPLoss(inverse=True)
                else:
                    loss_fn = CLIPLoss(inverse=False)
                video_recorder = self.record_video(self.fps, output_dir, f"{word_desciptor}_{phase}")
                model = Model().to(self.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                pbar = tqdm(range(self.total_steps))
                for _ in pbar:
                    optimizer.zero_grad()
                    parameters = model()
                    loss = 0
                    rend_imgs = []
                    for angle in self.view_angles:
                        temp_loss, temp_rendered_img = self.loss(
                            parameters, loss_fn=loss_fn, text=encoded_text, angle=angle
                        )
                        loss += temp_loss
                        rend_imgs.append(temp_rendered_img)
                    loss.backward()
                    optimizer.step()
                    pbar.set_description(f"Loss: {loss.item():.4f}")
                    collage = self.get_collage(rend_imgs)
                    if self.display:
                        cv2.imshow("image", cv2.resize(collage.copy(), (1024, 512)))
                        cv2.waitKey(1)
                    img_for_vid = np.clip((collage * 255), 0, 255).astype(np.uint8)
                    video_recorder.write(img_for_vid)

                video_recorder.release()
                cv2.destroyAllWindows()
                model_weights = model().detach().cpu().numpy()
                np.save(out_file_path, model_weights)


@hydra.main(config_path="../config", config_name="optimize")
def main(cfg: DictConfig):
    optimization = Optimization(**cfg.optimization_cfg)
    optimization.optimize()


if __name__ == "__main__":
    main()
