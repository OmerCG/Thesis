import json
import torch
from pathlib import Path
from typing import Literal, Dict, List
from torch.utils.data import Dataset


class CLIP2MESHDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        optimize_features: List[str],
    ):

        self.data_dir = data_dir
        self.optimize_features = optimize_features
        self.files = [file for file in Path(data_dir).rglob("*_labels.json")]
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
        for feature in self.optimize_features:
            if feature not in dict:
                raise ValueError(f"Feature {feature} not in dict {dict}")
            feature_data = dict[feature][0]
            if feature_data.__len__() > 10:
                feature_data = feature_data[:10]
            parameters_tensor = torch.cat((parameters_tensor, torch.tensor(feature_data)[None]))
        return parameters_tensor

    def labels_dict_to_tensor(self, dict: Dict[str, List[List[float]]]) -> torch.Tensor:
        return torch.tensor(list(dict.values()))[..., 0, 0]
