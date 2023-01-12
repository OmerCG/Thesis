import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing
from clip2mesh.utils import Utils, Pytorch3dRenderer


utils = Utils()


def adjust_rendered_img(img: torch.Tensor) -> np.ndarray:
    img = np.clip(img.cpu().numpy()[0, ..., :3] * 255, 0, 255).astype(np.uint8)
    return img


tgt_mesh_shape_vec = torch.tensor(
    np.load(
        "/home/nadav2/dev/data/CLIP2Shape/outs/vertices_heatmap/optimizations/smplx_multiview/long legs/long legs.npy"
    )
)

tgt_mesh_features = utils.get_smplx_model(betas=tgt_mesh_shape_vec, gender="male")
src_mesh_features = utils.get_smplx_model()

renderer = Pytorch3dRenderer(dist=2.4, elev=8.6, azim=2.1, img_size=(512, 512), texture_optimization=True)
tgt_mesh = renderer.get_mesh(*tgt_mesh_features)
src_mesh = renderer.get_mesh(*src_mesh_features)


src_verts = src_mesh.verts_packed()[None]
tgt_verts = tgt_mesh.verts_packed()[None]
# require gradients for target vertices
tgt_verts.requires_grad = True

# Compute chamfer distance
chamfer_dist = chamfer_distance(src_verts, tgt_verts)[0]
chamfer_dist.backward()


# display the gradient of the target vertices
normalized_grads = tgt_verts.grad / tgt_verts.grad.max()
color_map = plt.get_cmap("YlOrRd")

# vertex_colors = color_map(normalized_grads.detach().cpu().numpy())[..., :3]
vertex_colors = color_map((normalized_grads * 100.0).detach().cpu().numpy())[:, :, 0, :3]
vertex_colors = torch.tensor(vertex_colors).to(tgt_verts.device)

imgs_collage = []
for azim in np.arange(0, 360, 90):
    # create a new mesh with vertex colors
    rend_img = renderer.render_mesh(
        verts=tgt_verts.detach(),
        faces=torch.tensor(tgt_mesh_features[1]).cuda()[None].float(),
        texture_color_values=vertex_colors.float(),
        rotate_mesh={"degrees": azim, "axis": "y"},
    )
    rend_img = adjust_rendered_img(rend_img)
    imgs_collage.append(rend_img)

# concatenate the images to 2x2 collage
imgs_collage_1 = np.concatenate(imgs_collage[:2], axis=1)
imgs_collage_2 = np.concatenate(imgs_collage[2:], axis=1)
imgs_collage = np.concatenate([imgs_collage_1, imgs_collage_2], axis=0)
plt.imshow(imgs_collage)


imgs_collage = []
for verts_type in [tgt_verts.detach(), src_verts]:
    color = vertex_colors.float() if (verts_type == tgt_verts).all() else None
    rend_img = renderer.render_mesh(
        verts=verts_type,
        faces=torch.tensor(tgt_mesh_features[1]).cuda()[None].float(),
        texture_color_values=color,
        rotate_mesh={"degrees": 0, "axis": "y"},
    )
    rend_img = adjust_rendered_img(rend_img)
    cv2.putText(
        rend_img,
        "Source" if (verts_type == src_verts).all() else "Target",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )
    imgs_collage.append(rend_img)

imgs_collage_1 = np.concatenate(imgs_collage, axis=1)
