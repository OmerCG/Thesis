import cv2
import clip
import h5py
import torch
import hydra
import random
import numpy as np
from torch import nn
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from copy import deepcopy
from omegaconf import DictConfig
from pytorch3d.io import save_obj
from typing import Union, Any, Dict, Tuple, Literal, List
from clip2mesh.utils import Utils, Pytorch3dRenderer


class Image2Shape:
    def __init__(self, args):
        self.data_dir: Path = Path(args.data_dir)
        self.output_path: Path = Path(args.output_path)
        self.mode: Literal["video", "image"] = args.mode
        self.utils: Utils = Utils(comparison_mode=True)
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type: Literal["smplx", "flame", "smal"] = args.model_type
        self.with_face: bool = args.with_face
        self.suffix: bool = args.suffix
        self.save_obj: bool = args.save_obj
        self.with_orig_image: bool = args.with_orig_image
        self.save_orig_name: bool = args.save_orig_name
        self.verbose: bool = args.verbose
        self.comparison_mode: bool = args.comparison_mode
        self.display: bool = args.display_images
        self.num_img: int = 0  # num of collages saved
        self.collage: bool = args.collage

        if self.collage:
            self.images_collage_list = []
        if "body_pose_path" in args and args.body_pose_path is not None:
            self._load_body_pose(args.body_pose_path)

        self._load_renderer(args.renderer_kwargs)
        self._load_smplx_models(**args.smplx_models_paths)
        self._load_weights(args.labels_weights)
        self._load_clip_model()
        self._encode_labels()
        self._load_images_generator()

        if self.comparison_mode and args.comparison_data_path is not None:
            self._load_comparison_data(args.comparison_data_path)

    def _load_renderer(self, kwargs: Union[DictConfig, Dict[str, Any]]):
        self.renderer: Pytorch3dRenderer = Pytorch3dRenderer(**kwargs)

    def _load_body_pose(self, body_pose_path: str):
        self.body_pose: torch.Tensor = torch.from_numpy(np.load(body_pose_path))

    def _load_smplx_models(self, smplx_male: str, smplx_female: str):
        smplx_female, labels_female = self.utils.get_model_to_eval(smplx_female)
        smplx_male, labels_male = self.utils.get_model_to_eval(smplx_male)
        labels_female = self._flatten_list_of_lists(labels_female)
        labels_male = self._flatten_list_of_lists(labels_male)
        self.model: Dict[str, nn.Module] = {"male": smplx_male, "female": smplx_female}
        self.labels: Dict[str, List[str]] = {"male": labels_male, "female": labels_female}

    def _load_weights(self, labels_weights: Dict[str, float]):
        self.labels_weights = {}
        for gender, weights in labels_weights.items():
            self.labels_weights[gender] = (
                torch.tensor(weights).to(self.device)
                if weights is not None
                else torch.ones(len(self.labels[gender])).to(self.device)
            )

    def _from_h5_to_img(
        self, h5_file_path: Union[str, Path], gender: Literal["male", "female", "neutral"], renderer: Pytorch3dRenderer
    ) -> Tuple[np.ndarray, torch.Tensor]:
        data = h5py.File(h5_file_path, "r")
        shape_vector = torch.tensor(data["betas"])[None].float()
        render_mesh_kwargs = self.get_render_mesh_kwargs(shape_vector, gender=gender)
        rendered_img = renderer.render_mesh(**render_mesh_kwargs)
        rendered_img = self.adjust_rendered_img(rendered_img)
        return rendered_img, shape_vector

    def _load_comparison_data(self, path: Union[str, Path]):
        self.comparison_data: torch.Tensor = torch.from_numpy(np.load(path))

    def _load_images_generator(self):
        self.images_generator: List[Path] = sorted(list(self.data_dir.rglob(f"*.{self.suffix}")))

    def _load_clip_model(self):
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

    def _encode_labels(self):
        self.encoded_labels: Dict[str, torch.Tensor] = {
            gender: clip.tokenize(self.labels[gender]).to(self.device) for gender in self.labels.keys()
        }

    @staticmethod
    def _flatten_list_of_lists(list_of_lists: List[List[str]]) -> List[str]:
        return [item for sublist in list_of_lists for item in sublist]

    def _get_smplx_attributes(
        self, pred_vec: torch.Tensor, gender: Literal["male", "female", "neutral"]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        betas = pred_vec.cpu()
        if hasattr(self, "body_pose"):
            body_pose = self.body_pose
        else:
            body_pose = None
        smplx_out = self.utils.get_smplx_model(betas=betas, gender=gender, body_pose=body_pose)
        return smplx_out

    def _get_flame_attributes(
        self, pred_vec: torch.Tensor, gender: Literal["male", "female", "neutral"]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.with_face:
            flame_out = self.utils.get_flame_model(expression_params=pred_vec.cpu(), gender=gender)
        else:
            flame_out = self.utils.get_flame_model(shape_params=pred_vec.cpu(), gender=gender)
        return flame_out

    def _get_smal_attributes(self, pred_vec: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, None, None]:
        smal_out = self.utils.get_smal_model(beta=pred_vec.cpu())
        return smal_out

    def get_render_mesh_kwargs(
        self, pred_vec: torch.Tensor, gender: Literal["male", "female", "neutral"]
    ) -> Dict[str, np.ndarray]:
        out = self._get_smplx_attributes(pred_vec=pred_vec, gender=gender)

        kwargs = {"verts": out[0], "faces": out[1], "vt": out[2], "ft": out[3]}

        return kwargs

    def normalize_scores(self, scores: torch.Tensor, gender: Literal["male", "female", "neutral"]) -> torch.Tensor:
        normalized_score = scores * self.labels_weights[gender]
        return normalized_score.float()

    @staticmethod
    def adjust_rendered_img(img: torch.Tensor) -> np.ndarray:
        img = np.clip(img.cpu().numpy()[0, ..., :3] * 255, 0, 255).astype(np.uint8)
        return img

    def create_video_from_dir(self, dir_path: Union[str, Path], image_shape: Tuple[int, int]):
        dir_path = Path(dir_path)
        out_vid_path = dir_path.parent / "out_vid.mp4"
        out_vid = cv2.VideoWriter(out_vid_path.as_posix(), cv2.VideoWriter_fourcc(*"mp4v"), 30, image_shape)
        sorted_frames = sorted(dir_path.iterdir(), key=lambda x: int(x.stem))
        for frame in tqdm(sorted_frames, desc="Creating video", total=len(sorted_frames)):
            out_vid.write(cv2.imread(frame.as_posix()))
        out_vid.release()

    def _save_images_collage(self, images: List[np.ndarray]):
        collage_shape = self.utils.get_plot_shape(len(images))[0]
        images_collage = []
        for i in range(collage_shape[0]):
            images_collage.append(np.hstack(images[i * collage_shape[1] : (i + 1) * collage_shape[1]]))
        images_collage = np.vstack([image for image in images_collage])
        cv2.imwrite(self.images_out_path.as_posix(), images_collage)
        self.num_img += 1
        self.images_out_path = self.images_out_path.parent / f"{self.num_img}.png"

    def image_to_shape(self):
        if len(self.images_generator) < self.max_images_in_collage:
            self.max_images_in_collage = len(self.images_generator) - 1
        random.shuffle(self.images_generator)

        if not self.display:
            self.images_generator = tqdm(
                self.images_generator, desc="Rendering images", total=len(self.images_generator)
            )

        for idx, image_path in enumerate(self.images_generator):

            gender = image_path.stem.split("-")[0]

            image = Image.open(image_path.as_posix())
            encoded_image = self.clip_preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                if self.comparison_mode:
                    shape_vector_pred = self.comparison_data[idx][None]
                else:
                    clip_scores = self.clip_model(encoded_image, self.encoded_labels[gender])[0]
                    clip_scores = self.normalize_scores(clip_scores, gender=gender)
                    shape_vector_pred = self.model[gender](clip_scores)

            render_mesh_kwargs = self.get_render_mesh_kwargs(shape_vector_pred, gender=gender)

            rendered_img = self.renderer.render_mesh(**render_mesh_kwargs)
            rendered_img = self.adjust_rendered_img(rendered_img)

            input_image = cv2.imread(image_path.as_posix())
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            input_image = cv2.resize(input_image, rendered_img.shape[:2][::-1])

            if self.with_orig_image:
                concatenated_img = np.concatenate((input_image, np.array(rendered_img)), axis=1)
            else:
                concatenated_img = np.array(rendered_img)
            concatenated_img = cv2.cvtColor(concatenated_img, cv2.COLOR_BGR2RGB)

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
                        img_name = f"{self.num_img}_{gender}.png"
                    self.images_out_path = self.images_out_path.parent / img_name
                    cv2.imwrite(self.images_out_path.as_posix(), concatenated_img)
                    if self.save_obj:
                        save_obj(
                            self.images_out_path.as_posix().replace(f".{self.suffix}", ".obj"),
                            verts=torch.tensor(render_mesh_kwargs["verts"]),
                            faces=torch.tensor(render_mesh_kwargs["faces"]),
                        )

    def video_to_shape(self):
        for folder in self.data_dir.iterdir():

            if folder.is_dir():

                print(f"Processing {folder.name}...")

                dir_output_path = self.output_path / folder.name
                dir_output_path.mkdir(parents=True, exist_ok=True)

                images_output_path = dir_output_path / "images"
                images_output_path.mkdir(parents=True, exist_ok=True)

                if (images_output_path.parent / "out_vid.mp4").exists():
                    print(f"Video already exists for {folder.name}...")
                    continue
                gender = folder.name.split("-")[0]
                print(f"The gender is {gender}")

                renderer_kwargs = deepcopy(self.renderer_kwargs)
                renderer_kwargs.update({"tex_path": (folder / f"tex-{folder.name}.jpg").as_posix()})
                renderer = self._load_renderer(renderer_kwargs)

                gt_mesh_image, gt_shape_tensor = self._from_h5_to_img(
                    folder / "reconstructed_poses.hdf5", gender=gender, renderer=renderer
                )
                video_path = folder / (folder.name + ".mp4")
                video = cv2.VideoCapture(str(video_path))
                frame_counter = 0

                while video.isOpened():
                    ret, frame = video.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = Image.fromarray(frame)
                        encoded_image = self.clip_preprocess(frame).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            clip_scores = self.clip_model(encoded_image, self.encoded_labels[gender])[0]
                            clip_scores = self.normalize_scores(clip_scores, gender)
                            pred_shape_tensor = self.model[gender](clip_scores)

                        # calculate the l2 loss between the predicted and the ground truth shape
                        l2_loss = F.mse_loss(pred_shape_tensor.cpu(), gt_shape_tensor)

                        pred_mesh_kwargs = self.get_render_mesh_kwargs(pred_shape_tensor, gender=gender)
                        pred_mesh_img = renderer.render_mesh(**pred_mesh_kwargs)
                        pred_mesh_img = self.adjust_rendered_img(pred_mesh_img)

                        resized_frame = cv2.resize(np.array(frame), pred_mesh_img.shape[:2][::-1])

                        cv2.putText(resized_frame, "orig", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv2.putText(
                            resized_frame,
                            f"loss: {l2_loss.item():.4f}",
                            (0, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                        )
                        cv2.putText(pred_mesh_img, "pred", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv2.putText(gt_mesh_image, "gt", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                        concatenated_img = np.concatenate((resized_frame, pred_mesh_img, gt_mesh_image), axis=1)
                        concatenated_img = cv2.cvtColor(concatenated_img, cv2.COLOR_RGB2BGR)

                        cv2.imwrite(str(images_output_path / f"{frame_counter}.png"), concatenated_img)

                        if self.display:
                            cv2.putText(
                                concatenated_img,
                                f"frame: {frame_counter}",
                                (40, 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),
                                1,
                            )
                            cv2.imshow("frame", concatenated_img)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break

                        frame_counter += 1

                    else:
                        break

                video.release()
                cv2.destroyAllWindows()

                if self.create_vid:
                    self.create_video_from_dir(images_output_path, concatenated_img.shape[:2][::-1])

        def __call__(self):
            if self.mode == "image":
                self.image_to_shape()
            elif self.mode == "video":
                self.video_to_shape()


@hydra.main(config_path="../config", config_name="image_to_shape")
def main(args: DictConfig):
    image2shape = Image2Shape(args)
    image2shape()


if __name__ == "__main__":
    main()
