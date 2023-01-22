import math
import json
import hydra
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig
from itertools import permutations
from typing import Union, Dict, Any, Literal
from clip2mesh.utils import Utils
from clip2mesh.data_management.data_observation.choosing_descriptors import ChoosingDescriptors


class ChoosingDescriptorsArik(ChoosingDescriptors):
    def __init__(
        self,
        images_dir: Union[Path, str],
        max_num_of_descriptors: int,
        min_num_of_descriptors: int,
        descriptors_clusters_json: str,
        corr_threshold: float = 0.5,
        iou_threshold: float = 0.7,
        output_dir: Union[Path, str] = None,
    ):
        super().__init__()
        self.utils = Utils()
        self.corr_threshold = corr_threshold
        self.iou_threshold = iou_threshold
        self.images_dir: Path = Path(images_dir)
        self.max_num_of_descriptors = max_num_of_descriptors
        self.min_num_of_descriptors = min_num_of_descriptors
        if output_dir is not None:
            self.output_dir: Path = Path(output_dir)
        if descriptors_clusters_json is not None:
            self.clusters = self.get_clusters(Path(descriptors_clusters_json))
        self._get_logger()

    def _get_logger(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s: - %(message)s")
        self.logger = logging.getLogger("choosing_descriptors")

    def get_dfs(self, jsons_dir: Union[Path, str]):
        json_files = list(Path(jsons_dir).rglob("*_labels.json"))
        df = pd.DataFrame()
        for json_file in tqdm(json_files, desc="Loading json files", total=len(json_files)):
            with open(json_file, "r") as f:
                json_data = json.load(f)
                df = pd.concat([df, pd.DataFrame(json_data)], axis=0)
        df = df.apply(lambda x: [y[0] for y in x])

        if "feminine" in df.columns:
            df = df.drop("feminine", axis=1)

        # get variances
        variances = df.var(axis=0)
        variances.sort_values(ascending=False, inplace=True)
        variances = pd.DataFrame(zip(variances.index, variances.values), columns=["descriptor", "variance"])

        # get corrlation matrix between descriptors
        corr_df = pd.DataFrame(columns=["descriptor_1", "descriptor_2", "correlation"])
        permut_list = []
        for perm in permutations(df.columns, 2):
            permut_list.append(perm)
        for perm in tqdm(permut_list, desc="Calculating correlations", total=len(permut_list)):
            corr_df = pd.concat(
                [
                    corr_df,
                    pd.DataFrame(
                        {
                            "descriptor_1": [perm[0]],
                            "descriptor_2": [perm[1]],
                            "correlation": [np.corrcoef(df[perm[0]], df[perm[1]])[0, 1]],
                        }
                    ),
                ],
                axis=0,
            )
        return corr_df, variances

    @staticmethod
    def get_descriptor_iou(descriptor: str, ious_df: pd.DataFrame) -> float:
        return ious_df[(ious_df["descriptor_1"] == descriptor) | (ious_df["descriptor_2"] == descriptor)]["iou"].mean()

    @staticmethod
    def get_clusters(descriptors_clusters_json: Path):
        with open(descriptors_clusters_json, "r") as f:
            clusters = json.load(f)
        return clusters

    def check_if_descriptor_is_correlated_with_chosen_descriptors(
        self, descriptor: str, chosen_descriptors: Dict[str, Dict[str, Any]], correlations_df: pd.DataFrame
    ) -> bool:
        for chosen_descriptor in chosen_descriptors:
            descriprtor_correlations = correlations_df[
                (
                    (correlations_df["descriptor_1"] == descriptor)
                    & (correlations_df["descriptor_2"] == chosen_descriptor)
                    # | (correlations_df["descriptor_1"] == chosen_descriptor)
                    # & (correlations_df["descriptor_2"] == descriptor)
                )
            ]
            if abs(descriprtor_correlations["correlation"][0]) > self.corr_threshold:
                return True, chosen_descriptor, descriprtor_correlations["correlation"][0]
        return False, None, None

    def candidate_iou_with_chosen_descriptors(
        self, descriptor: str, chosen_descriptors: Dict[str, Dict[str, Any]], ious_df: pd.DataFrame
    ) -> float:
        for chosen_descriptor in chosen_descriptors:
            iou = ious_df[
                ((ious_df["descriptor_1"] == descriptor) & (ious_df["descriptor_2"] == chosen_descriptor))
                | ((ious_df["descriptor_1"] == chosen_descriptor) & (ious_df["descriptor_2"] == descriptor))
            ]["iou"].values[0]
            if iou > self.iou_threshold:
                return False, chosen_descriptor, iou
        return True, None, None

    def my_test(self):
        # get the correlations and variances of the data
        correlations_df, variances = self.get_dfs(self.images_dir)
        self.logger.info(f"There are total of {len(variances)} descriptors")

        # get the clusters
        if hasattr(self, "clusters"):
            chosen_descriptors = {cluster_id: {} for cluster_id in self.clusters}
            # add column of cluster to summary df
            variances["cluster"] = variances["descriptor"].apply(
                lambda x: self.find_cluster_of_descriptor(x, self.clusters)
            )
        else:
            chosen_descriptors = {0: {}}
            variances["cluster"] = np.zeros(len(variances)).astype(int)

        # start iterating over the clusters
        for cluster, cluster_df in variances.groupby("cluster"):

            self.logger.info(f"Working on cluster {cluster}")

            # sort the cluster df by variance
            cluster_df = cluster_df.sort_values(by="variance", ascending=False)

            # iterate over the cluster descriptors
            for i, (_, row) in enumerate(cluster_df.iterrows()):

                # get the descriptor and its variance
                descriptor = row["descriptor"]
                descriptor_var = row["variance"]

                self.logger.info(f"{i} most variance descriptor - {descriptor}")

                # if this is the descriptor with the highest variance - add it
                if chosen_descriptors[cluster] == {}:

                    self.logger.info(f"Adding {row['descriptor']} to chosen descriptors")

                    descriptor = row["descriptor"]

                    chosen_descriptors[cluster][descriptor] = {
                        "variance": descriptor_var,
                    }

                else:

                    # check if the descriptor is correlated with the chosen descriptors
                    corr_test = self.check_if_descriptor_is_correlated_with_chosen_descriptors(
                        descriptor, chosen_descriptors[cluster], correlations_df
                    )
                    if not corr_test[0]:

                        self.logger.info(f"Adding {row['descriptor']} to chosen descriptors")

                        chosen_descriptors[cluster][descriptor] = {
                            "variance": descriptor_var,
                        }

                    else:

                        self.logger.info(
                            f"{row['descriptor']} is correlated with {corr_test[1]} with corr {corr_test[2]}"
                        )

            self.logger.info(f"currently chosen descriptors: {chosen_descriptors[cluster]}")
            self.logger.info(f"Finished cluster {cluster}")
            print("*" * 100)

        self.logger.info(
            f"There are {sum([len(descriptors) for descriptors in chosen_descriptors.values()])} chosen descriptors in initial filter"
        )
        self.logger.info(f"chosen descriptors: {[list(chosen_descriptors[x].keys()) for x in chosen_descriptors]}")
        return chosen_descriptors, correlations_df, variances

    def final_filter(self, chosen_descriptors: Dict[str, Dict[str, float]], correlations_df: pd.DataFrame):
        self.logger.info("Final filter")
        chosen_descriptors_df = pd.DataFrame(self.flatten_dict_of_dicts(chosen_descriptors)).T.sort_values(
            by="variance", ascending=False
        )
        subset_correlation_df = correlations_df[(correlations_df["descriptor_1"].isin(chosen_descriptors_df.index))]

        removed_descriptors = []
        chosed_descriptors = []
        for descriptor, data in chosen_descriptors_df.iterrows():
            if descriptor in removed_descriptors:
                continue

            descriptor_cluster = self.find_cluster_of_descriptor(descriptor, chosen_descriptors)

            self.logger.info(f"Descriptor {descriptor} has variance {data['variance']}")
            for cluster_id in chosen_descriptors.keys():
                if cluster_id == descriptor_cluster:
                    continue
                corr_df = subset_correlation_df[
                    (subset_correlation_df["descriptor_1"] == descriptor)
                    & subset_correlation_df["descriptor_2"].isin(chosen_descriptors[cluster_id])
                ]
                high_corr_df = corr_df[corr_df["correlation"] > self.corr_threshold]
                if high_corr_df.index.__len__() > 0:

                    # get all descriptors that are correlated with the current descriptor
                    corr_descriptors = high_corr_df["descriptor_2"].values
                    for corr_descriptor, corr_value in zip(corr_descriptors, high_corr_df["correlation"].values):
                        self.logger.info(
                            f"Descriptor {descriptor} is correlated with {corr_descriptor} with correlation {corr_value} - removing {corr_descriptor}"
                        )
                        if corr_descriptor not in removed_descriptors and corr_descriptor not in chosed_descriptors:
                            removed_descriptors.append(corr_descriptor)
                if descriptor not in chosed_descriptors and descriptor not in removed_descriptors:
                    chosed_descriptors.append(descriptor)
        removed_descriptors_df = chosen_descriptors_df[chosen_descriptors_df.index.isin(removed_descriptors)]
        chosen_descriptors_df = chosen_descriptors_df.drop(removed_descriptors)
        return chosen_descriptors_df, removed_descriptors_df

    def choose(self):
        chosen_descriptors, correlations_df, variance_df = self.my_test()
        final_filtered_chosen_descriptors, removed_descriptors_df = self.final_filter(
            chosen_descriptors, correlations_df
        )
        final_choose = {cluster_id: {} for cluster_id in self.clusters}
        for desc in final_filtered_chosen_descriptors.index:
            final_choose[self.find_cluster_of_descriptor(desc, chosen_descriptors)][
                desc
            ] = final_filtered_chosen_descriptors.loc[desc]["variance"]
        number_of_descriptors = self.get_num_of_chosen_descriptors(final_choose)
        if number_of_descriptors > self.max_num_of_descriptors:
            self.logger.info(f"Too many descriptors ({number_of_descriptors}) - reducing")
            while number_of_descriptors > self.max_num_of_descriptors:
                final_choose = self.reduce_descriptor(final_choose)
                number_of_descriptors = self.get_num_of_chosen_descriptors(final_choose)
        elif number_of_descriptors < self.min_num_of_descriptors:
            iterator = 0
            while number_of_descriptors < self.min_num_of_descriptors:
                if iterator < removed_descriptors_df.shape[0]:
                    cluster_id = self.find_cluster_of_descriptor(
                        removed_descriptors_df.iloc[iterator].name, self.clusters
                    )
                    (final_choose[cluster_id]).update(
                        {removed_descriptors_df.iloc[iterator].name: removed_descriptors_df.iloc[iterator].variance}
                    )
                    number_of_descriptors = self.get_num_of_chosen_descriptors(final_choose)
                    iterator += 1
                else:
                    for _, row in variance_df.iterrows():
                        if row.descriptor not in self.flatten_dict_of_dicts(final_choose):
                            cluster_id = self.find_cluster_of_descriptor(row.descriptor, self.clusters)
                            final_choose[cluster_id][row.descriptor] = row.variance
                            number_of_descriptors = self.get_num_of_chosen_descriptors(final_choose)
                            if number_of_descriptors >= self.min_num_of_descriptors:
                                break

        if hasattr(self, "output_dir"):
            with open(self.output_dir / f"chosen_descriptors.json", "w") as f:
                json.dump(final_choose, f)

        return final_choose


@hydra.main(config_path="../../config", config_name="choose_algorithm_arik")
def main(cfg: DictConfig) -> None:
    choosing_descriptors = ChoosingDescriptorsArik(**cfg)
    final_choose = choosing_descriptors.choose()
    print(f"Chosen descriptors: {final_choose}")


if __name__ == "__main__":
    main()
