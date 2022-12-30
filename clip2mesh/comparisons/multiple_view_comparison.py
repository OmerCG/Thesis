import cv2
import torch
import numpy as np
from pathlib import Path
from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes
from typing import Dict, Union
from clip2mesh.utils import Pytorch3dRenderer, Utils


def get_kwargs_from_mesh(mesh: Meshes) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    return {"verts": mesh.verts_packed(), "faces": mesh.faces_packed()}


def adjust_rendered_img(img: torch.Tensor):
    img = np.clip(img.cpu().numpy()[0, ..., :3] * 255, 0, 255).astype(np.uint8)
    return img


utils = Utils(comparison_mode=True)
input_dir = Path("/home/nadav2/dev/data/CLIP2Shape/outs/images_to_shape/smplx_shape/shapy_cherry_pick/comparison")

for person_dir in input_dir.iterdir():

    orig_image = cv2.imread(str(person_dir / "orig.png"))
    pred_obj = str(person_dir / f"{person_dir.name}_pred.obj")
    shapy_obj = str(person_dir / f"{person_dir.name}_shapy.obj")

    pred_mesh = load_objs_as_meshes([pred_obj], device="cuda")
    shapy_mesh = load_objs_as_meshes([shapy_obj], device="cuda")

    renderer_kwargs = {
        "img_size": list(orig_image.shape[:2]),
        "tex_path": None,
        "azim": 15.0,
        "dist": 2.7,
        "elev": 15.0,
    }

    renderer = Pytorch3dRenderer(**renderer_kwargs)

    pred_img = renderer.render_mesh(**get_kwargs_from_mesh(pred_mesh))
    shapy_img = renderer.render_mesh(**get_kwargs_from_mesh(shapy_mesh))

    pred_img = adjust_rendered_img(pred_img)
    shapy_img = adjust_rendered_img(shapy_img)
