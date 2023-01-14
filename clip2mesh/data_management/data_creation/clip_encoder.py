import clip
import json
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from typing import List
from pathlib import Path
from clip2mesh.utils import Utils


def generate_clip_scores(device: str, side: bool, imgs_dir: str, labels: List[List[str]] = None):

    utils = Utils()
    model, preprocess = clip.load("ViT-B/32", device=device)

    if labels is None:
        labels = utils.get_labels()
    if side:
        files_generator = sorted(list(Path(imgs_dir).rglob("*front.png")), key=lambda x: int(x.stem.split("_")[0]))
    else:
        files_generator = sorted(list(Path(imgs_dir).rglob("*.png")), key=lambda x: int(x.stem.split("_")[0]))
    dir_length = len(files_generator)
    encoded_labels = {label[0]: clip.tokenize(label).to(device) for label in labels}

    for file in tqdm(files_generator, desc="generating clip scores", total=dir_length):

        json_path = file.parent / f"{file.stem.split('_')[0]}_labels.json"
        json_data = {}

        encoded_frontal_image = preprocess(Image.open(file.as_posix())).unsqueeze(0).to(device)
        if side:
            try:
                encoded_side_image = (
                    preprocess(Image.open((file.parent / file.name.replace("front", "side")).as_posix()))
                    .unsqueeze(0)
                    .to(device)
                )
            except FileNotFoundError:
                print(f"Side image not found for {file.name}")
                continue

        with torch.no_grad():

            # get the mean value of the front and side images for each label
            for label, encoded_label in encoded_labels.items():
                front_score = model(encoded_frontal_image, encoded_label)[0].cpu().numpy()
                if side:
                    side_score = model(encoded_side_image, encoded_label)[0].cpu().numpy()
                    json_data[label] = ((front_score + side_score) / 2).tolist()
                else:
                    json_data[label] = front_score.tolist()

        with open(json_path, "w") as f:
            json.dump(json_data, f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("imgs_dir", type=str)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-s", "--side", action="store_true", default=False)
    return parser.parse_args()


def main(imgs_dir: str, device: str, side: bool):
    generate_clip_scores(device, side, imgs_dir)


if __name__ == "__main__":
    kwargs = parse_args()
    main(**vars(kwargs))
