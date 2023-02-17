import cv2
import torch
import hydra
import numpy as np
from PIL import Image
from pathlib import Path
from omegaconf import DictConfig
from typing import Dict, Tuple, Literal
from clip2mesh.utils import Utils, Image2ShapeUtils


class Image2Shape(Image2ShapeUtils):
    def __init__(
        self,
        smplx_model: str,
        output_path: str,
        renderer_kwargs: Dict[str, Dict[str, float]],
    ):
        super().__init__()
        self.output_path: Path = Path(output_path)
        self.utils: Utils = Utils(comparison_mode=True)
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        self._load_renderer(renderer_kwargs)
        self.load_smplx_model(smplx_model)
        self._load_clip_model()
        self._encode_labels()

    def load_smplx_model(self, model_path: str):
        self.model, labels = self.utils.get_model_to_eval(model_path)
        self.labels = self._flatten_list_of_lists(labels)

    def get_body_shapes(
        self, raw_img_path: Path, gender: Literal["male", "female", "neutral"]
    ) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:

        # load raw image
        raw_img = cv2.imread(str(raw_img_path))
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        # clip preprocess
        encoded_image = self.clip_preprocess(Image.fromarray(raw_img)).unsqueeze(0).to(self.device)

        # our prediction
        with torch.no_grad():
            clip_scores = self.clip_model(encoded_image, self.encoded_labels)[0].float()
            our_body_shape = self.model(clip_scores).cpu()

        return our_body_shape, raw_img

    def get_rendered_images(
        self, smplx_features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], angle: float
    ) -> Dict[str, np.ndarray]:
        """Render the meshes for the different methods"""
        kwargs = {"rotate_mesh": {"degrees": float(angle), "axis": "y"}}
        rendered_img = self.renderer.render_mesh(*smplx_features, **kwargs)
        rendered_img = self.adjust_rendered_img(rendered_img)
        return rendered_img

    def __call__(self, image_path: Path, gender: Literal["male", "female", "neutral"]):

        # get body shapes
        our_body_shape, raw_img = self.get_body_shapes(raw_img_path=Path(image_path), gender=gender)

        smplx_features = self._get_smplx_attributes(our_body_shape, gender)
        # get rendered images
        rendered_img = self.get_rendered_images(smplx_features, angle=20)
        # rendered_img = cv2.resize(rendered_img, raw_img.shape[::-1][1:])
        # concatenate images
        # concatenated_img = np.concatenate([raw_img, rendered_img], axis=1)
        # concatenated_img = cv2.cvtColor(concatenated_img, cv2.COLOR_RGB2BGR)
        # save image
        image_suffix = Path(image_path).suffix
        # cv2.imwrite(image_path.as_posix().replace(image_suffix, f"_out{image_suffix}"), rendered_img)
        cv2.imwrite("/home/nadav2/dev/data/CLIP2Shape/outs/images_from_demo/image2shape/out.png", rendered_img)


@hydra.main(config_path="../config", config_name="image2shape")
def main(cfg: DictConfig) -> None:
    image_path = Path("/home/nadav2/Downloads/image_for_teaser.jpg")
    hbw_comparison = Image2Shape(**cfg)
    hbw_comparison(image_path, "female")


if __name__ == "__main__":
    main()
