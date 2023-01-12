import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from clip2mesh.utils import Utils, Pytorch3dRenderer

utils = Utils()
debug = False
data = [
    file
    for file in Path("/home/nadav2/dev/data/CLIP2Shape/images/smplx_female_uniform_multiview").rglob("*.json")
    if "labels" not in file.stem
]
heights = []
for file in tqdm(data, total=len(data), desc="loading data..."):

    with open(file.as_posix(), "r") as f:
        data = json.load(f)
    betas = torch.tensor(data["betas"])
    verts, faces, vt, ft = utils.get_smplx_model(betas=betas, gender="female")
    verts = torch.tensor(verts).cuda()
    faces = torch.tensor(faces).cuda()
    renderer = Pytorch3dRenderer(dist=2.4, elev=8.6, azim=2.1, img_size=(512, 512), texture_optimization=True)
    min_height = verts.min(axis=0)
    max_height = verts.max(axis=0)
    height = max_height.values[1] - min_height.values[1]
    heights.append(height.item())

    if debug:
        # color vertices by height
        colors = torch.ones_like(verts) * 0.7
        min_idxs = verts.topk(dim=0, k=100, largest=False).indices[..., 1]
        max_idxs = verts.topk(dim=0, k=100, largest=True).indices[..., 1]
        colors[min_idxs.detach().cpu()] = torch.tensor([1, 0, 0]).expand(min_idxs.shape[0], 3).float().cuda()
        colors[max_idxs.detach().cpu()] = torch.tensor([0, 1, 0]).expand(max_idxs.shape[0], 3).float().cuda()
        renderered_img = renderer.render_mesh(verts[None], faces[None], texture_color_values=colors[None].cuda())
        img = np.clip(renderered_img.cpu().numpy()[0, ..., :3] * 255, 0, 255).astype(np.uint8)

plt.hist(heights, bins=100)
plt.title("Height Histogram")
plt.yticks([])
plt.show()
