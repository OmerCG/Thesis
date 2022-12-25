import cv2
import clip
import torch
import hydra
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from omegaconf import DictConfig
from pytorch3d.io import save_obj
from typing import Union, Any, Dict, Tuple
from clip2mesh.utils import Utils, Pytorch3dRenderer


class Image2Shape:
    def __init__(self, args, max_images_in_collage: int = 25):
        self.images_dir = Path(args.images_dir)
        self.utils = Utils()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = args.model_type
        self.with_face = args.with_face
        self.gender = args.gender
        self.suffix = args.suffix
        self.save_obj = args.save_obj
        self.with_orig_image = args.with_orig_image
        self.save_orig_name = args.save_orig_name
        self.verbose = args.verbose
        self.comparison_mode = args.comparison_mode
        self.display = args.display_images
        self.max_images_in_collage = max_images_in_collage
        self.num_img = 0  # num of collages saved
        self.collage = args.collage

        if args.images_out_path is not None:
            self.images_out_path = Path(args.images_out_path) / f"{self.num_img}.png"
            if self.collage:
                self.images_collage_list = []
        if args.labels_weights is not None:
            self.labels_weights = torch.tensor(args.labels_weights).to(self.device)
        if args.rest_pose_path is not None:
            self._load_body_pose(args.rest_pose_path)

        self._load_renderer(args.renderer_kwargs)
        self._load_model(args.model_path)
        self._load_clip_model()
        self._encode_labels()
        self._load_images_generator()

        if self.comparison_mode and args.comparison_data_path is not None:
            self._load_comparison_data(args.comparison_data_path)

    def _load_renderer(self, kwargs: Union[DictConfig, Dict[str, Any]]):
        self.renderer = Pytorch3dRenderer(**kwargs)
        self.image_size = kwargs.img_size

    def _load_body_pose(self, rest_pose_path: str):
        self.rest_pose = torch.from_numpy(np.load(rest_pose_path))

    def _load_model(self, model_path: str):
        self.model, labels = self.utils.get_model_to_eval(model_path)
        self.labels = self._flatten_list_of_lists(labels)
        if not hasattr(self, "labels_weights"):
            self.labels_weights = torch.ones(len(self.labels)).to(self.device)

    def _load_comparison_data(self, path):
        self.comparison_data = torch.from_numpy(np.load(path))

    def _load_images_generator(self):
        if self.comparison_mode:
            self.images_generator = sorted(list(self.images_dir.rglob(f"*.{self.suffix}")))
        else:
            self.images_generator = list(self.images_dir.rglob(f"*_{self.gender}.{self.suffix}"))

    def _load_clip_model(self):
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

    def _encode_labels(self):
        self.encoded_labels = clip.tokenize(self.labels).to(self.device)

    @staticmethod
    def _flatten_list_of_lists(list_of_lists):
        return [item for sublist in list_of_lists for item in sublist]

    def _get_smplx_attributes(self, pred_vec: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        betas = pred_vec.cpu()
        if hasattr(self, "rest_pose"):
            body_pose = self.rest_pose
        else:
            body_pose = None
        smplx_out = self.utils.get_smplx_model(betas=betas, gender=self.gender, body_pose=body_pose)
        return smplx_out

    def _get_flame_attributes(self, pred_vec: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.with_face:
            flame_out = self.utils.get_flame_model(expression_params=pred_vec.cpu(), gender=self.gender)
        else:
            flame_out = self.utils.get_flame_model(shape_params=pred_vec.cpu(), gender=self.gender)
        return flame_out

    def _get_smal_attributes(self, pred_vec: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        smal_out = self.utils.get_smal_model(beta=pred_vec.cpu())
        return smal_out

    def get_render_mesh_kwargs(self, pred_vec: torch.Tensor) -> Dict[str, np.ndarray]:
        if self.model_type == "smplx":
            out = self._get_smplx_attributes(pred_vec=pred_vec)
        elif self.model_type == "flame":
            out = self._get_flame_attributes(pred_vec=pred_vec)
        else:
            out = self._get_smal_attributes(pred_vec=pred_vec)

        kwargs = {"verts": out[0], "faces": out[1], "vt": out[2], "ft": out[3]}

        return kwargs

    def normalize_scores(self, scores: torch.Tensor) -> torch.Tensor:
        clip_min_value = 15
        clip_max_value = 30
        # normalized_score = (scores - clip_min_value) / (clip_max_value - clip_min_value)
        normalized_score = scores * self.labels_weights
        # normalized_score = torch.clamp(normalized_score, 12, 30)
        # normalized_score *= 40
        return normalized_score.float()

    @staticmethod
    def adjust_rendered_img(img: torch.Tensor):
        img = np.clip(img.cpu().numpy()[0, ..., :3] * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img)

    def _save_images_collage(self, images: list):
        collage_shape = self.utils.get_plot_shape(len(images))[0]
        images_collage = []
        for i in range(collage_shape[0]):
            images_collage.append(np.hstack(images[i * collage_shape[1] : (i + 1) * collage_shape[1]]))
        images_collage = np.vstack([image for image in images_collage])
        cv2.imwrite(self.images_out_path.as_posix(), images_collage)
        self.num_img += 1
        self.images_out_path = self.images_out_path.parent / f"{self.num_img}.png"

    def __call__(self):

        if len(self.images_generator) < self.max_images_in_collage:
            self.max_images_in_collage = len(self.images_generator) - 1
        random.shuffle(self.images_generator)

        if not self.display:
            self.images_generator = tqdm(
                self.images_generator, desc="Rendering images", total=len(self.images_generator)
            )

        for idx, image_path in enumerate(self.images_generator):
            image = Image.open(image_path.as_posix())
            encoded_image = self.clip_preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                if self.comparison_mode:
                    shape_vector_pred = self.comparison_data[idx][None]
                else:
                    clip_scores = self.clip_model(encoded_image, self.encoded_labels)[0]
                    clip_scores = self.normalize_scores(clip_scores)
                    shape_vector_pred = self.model(clip_scores)

            render_mesh_kwargs = self.get_render_mesh_kwargs(shape_vector_pred)

            rendered_img = self.renderer.render_mesh(**render_mesh_kwargs)
            rendered_img = self.adjust_rendered_img(rendered_img)

            input_image = cv2.imread(image_path.as_posix())
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            input_image = cv2.resize(input_image, self.image_size)

            if self.with_orig_image:
                concatenated_img = np.concatenate((input_image, np.array(rendered_img)), axis=1)
            else:
                concatenated_img = np.array(rendered_img)
            concatenated_img = cv2.cvtColor(concatenated_img, cv2.COLOR_BGR2RGB)

            # write clip scores on the image
            if self.verbose:
                for i, label in enumerate(self.labels):
                    cv2.putText(
                        concatenated_img,
                        f"{label}: {clip_scores[0][i].item():.2f}",
                        (370, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

            if self.display:
                cv2.imshow("input", concatenated_img)
                key = cv2.waitKey(0)
                if key == ord("q"):
                    break

            if hasattr(self, "images_out_path"):
                if self.collage:
                    if len(self.images_collage_list) == self.max_images_in_collage:
                        self._save_images_collage(self.images_collage_list)
                        self.images_collage_list = []
                    else:
                        self.images_collage_list.append(concatenated_img)
                else:
                    if self.save_orig_name:
                        img_name = image_path.name
                    else:
                        self.num_img += 1
                        img_name = f"{self.num_img}_{self.gender}.png"
                    self.images_out_path = self.images_out_path.parent / img_name
                    cv2.imwrite(self.images_out_path.as_posix(), concatenated_img)
                    if self.save_obj:
                        save_obj(
                            self.images_out_path.as_posix().replace(f".{self.suffix}", ".obj"),
                            verts=torch.tensor(render_mesh_kwargs["verts"]),
                            faces=torch.tensor(render_mesh_kwargs["faces"]),
                        )


@hydra.main(config_path="config", config_name="image_to_shape")
def main(args: DictConfig):
    image2shape = Image2Shape(args)
    image2shape()


if __name__ == "__main__":
    main()
