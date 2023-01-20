import json
import torch
from pathlib import Path
from typing import Dict, List
from torch.utils.data import Dataset


class CLIP2MESHDataset(Dataset):
    def __init__(self, data_dir: str, optimize_feature: str, labels_to_get: List[str], out_features: int = 10):

        self.data_dir = data_dir
        self.out_features = out_features
        self.optimize_feature = optimize_feature
        self.labels_to_get = labels_to_get
        self.files = sorted(
            [file for file in Path(data_dir).rglob("*_labels.json")], key=lambda x: int(x.stem.split("_")[0])
        )
        self.files.sort()

    def __len__(self):
        return self.files.__len__()

    def __getitem__(self, idx):
        # extract parameters
        parameters_json = self.files[idx].as_posix().replace("_labels.json", ".json")
        with open(parameters_json, "r") as f:
            parameters = json.load(f)

        parameters_t = self.params_dict_to_tensor(parameters)

        # extract labels
        clip_scores_json = self.files[idx]
        with open(clip_scores_json, "r") as f:
            clip_scores = json.load(f)

        labels = self.labels_dict_to_tensor(clip_scores)
        return parameters_t, labels

    def params_dict_to_tensor(self, dict: Dict[str, List[float]]) -> torch.Tensor:
        parameters_tensor = torch.tensor([])
        if self.optimize_feature not in dict:
            raise ValueError(f"Feature {self.optimize_feature} not in dict {dict}")
        feature_data = dict[self.optimize_feature][0]
        if feature_data.__len__() > self.out_features:
            feature_data = feature_data[: self.out_features]
        parameters_tensor = torch.cat((parameters_tensor, torch.tensor(feature_data)[None]))
        return parameters_tensor

    def labels_dict_to_tensor(self, dict: Dict[str, List[List[float]]]) -> torch.Tensor:
        return torch.tensor([dict[descriptor] for descriptor in self.labels_to_get])[..., 0, 0]

    def get_labels(self):
        return [[label] for label in self.labels_to_get]
