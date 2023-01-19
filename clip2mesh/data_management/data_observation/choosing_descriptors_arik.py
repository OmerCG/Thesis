import math
import json
import hydra
import logging
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig
from itertools import permutations
from typing import Union, Dict, Any, Literal
from clip2mesh.utils import Utils
from clip2mesh.data_management.data_observation.choosing_descriptors import ChoosingDescriptors


class ChoosingDescriptorsArik(ChoosingDescriptors):
    def __init__(self, args):
        super().__init__()
        self.utils = Utils()
        self.corr_threshold = 0.5
        self.iou_threshold = 0.7
        self.mode: Literal["an", "or"] = args.mode
        self.working_dir: Path = Path(args.working_dir)
        self.images_dir: Path = Path(args.images_dir)
        self.max_num_of_descriptors: int = args.max_num_of_descriptors
        self.min_num_of_descriptors: int = args.min_num_of_descriptors
        self.descriptors_clusters_json: Path = Path(args.descriptors_clusters_json)
        self.clusters = self.get_clusters()
        self._get_logger()

    def _get_logger(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s: - %(message)s")
        self.logger = logging.getLogger("choosing_descriptors")

    def get_corr_df(self, jsons_dir: Union[Path, str]):
        json_files = list(Path(jsons_dir).rglob("*_labels.json"))
        df = pd.DataFrame()
        for json_file in tqdm(json_files, desc="Loading json files", total=len(json_files)):
            with open(json_file, "r") as f:
                json_data = json.load(f)
                df = pd.concat([df, pd.DataFrame(json_data)], axis=0)
        df = df.apply(lambda x: [y[0] for y in x])
        permut_list = []
        for x, y in permutations(df.columns, 2):
            if (x, y) in permut_list or (y, x) in permut_list:
                continue
            permut_list.append((x, y))
        # get the correlation between the descriptors
        corr_df = pd.DataFrame()
        for x, y in tqdm(permut_list, desc="Calculating correlations", total=len(permut_list)):
            corr_df.loc[x, y] = df[x].corr(df[y])
        return corr_df

    @staticmethod
    def get_descriptor_iou(descriptor: str, ious_df: pd.DataFrame) -> float:
        return ious_df[(ious_df["descriptor_1"] == descriptor) | (ious_df["descriptor_2"] == descriptor)]["iou"].mean()

    def get_clusters(self):
        with open(self.descriptors_clusters_json, "r") as f:
            clusters = json.load(f)
        return clusters

    def check_if_descriptor_is_correlated_with_chosen_descriptors(
        self, descriptor: str, chosen_descriptors: Dict[str, Dict[str, Any]], correlations_df: pd.DataFrame
    ) -> bool:
        for chosen_descriptor in chosen_descriptors:
            try:
                corr = abs(correlations_df.loc[descriptor, chosen_descriptor])
                if math.isnan(corr):
                    raise KeyError
            except KeyError:
                corr = abs(correlations_df.loc[chosen_descriptor, descriptor])
                if corr > self.corr_threshold:
                    return True, chosen_descriptor, corr
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

    def and_mode(self):

        # get the df that contains the vertex coverage of each descriptor
        summary_df = pd.read_csv(self.working_dir / "vertex_heatmaps" / "summary.csv")

        self.logger.info(f"There are total of {len(summary_df)} descriptors")

        # get the correlation df
        correlations_df = self.get_corr_df(self.images_dir)

        # get the iou df
        ious_df = pd.read_csv(self.working_dir / "vertex_heatmaps" / "ious.csv")

        # get the clusters
        clusters = self.get_clusters()
        chosen_descriptors = {cluster_id: {} for cluster_id in clusters}

        # add column of cluster to summary df
        summary_df["cluster"] = summary_df["descriptor"].apply(lambda x: self.find_cluster_of_descriptor(x, clusters))

        # start iterating over the clusters
        for cluster, cluster_df in summary_df.groupby("cluster"):

            self.logger.info(f"Working on cluster {cluster}")

            # sort the cluster df by vertex coverage
            cluster_df = cluster_df.sort_values(by="vertex_coverage", ascending=False)

            # iterate over the cluster descriptors
            for i, (_, row) in enumerate(cluster_df.iterrows()):

                descriptor = row["descriptor"]
                self.logger.info(f"{i} most covering descriptor - {descriptor}")

                # if this is the descriptor with the highest vertex coverage - add it
                if chosen_descriptors[cluster] == {}:

                    self.logger.info(f"Adding {row['descriptor']} to chosen descriptors")

                    descriptor = row["descriptor"]

                    descriptor_avg_iou = self.get_descriptor_iou(descriptor, ious_df)

                    chosen_descriptors[cluster][descriptor] = {
                        "iou": descriptor_avg_iou,
                        "vertex_coverage": row["vertex_coverage"],
                    }

                else:

                    # get descriptor avg iou with all other descriptors in the total dataset
                    descriptor_avg_iou = self.get_descriptor_iou(descriptor, ious_df)

                    # check descriptor's iou with chosen descriptors. if it is above threshold - skip
                    iou_test = self.candidate_iou_with_chosen_descriptors(
                        descriptor, chosen_descriptors[cluster], ious_df
                    )

                    if iou_test[0]:

                        # check if the descriptor is correlated with the chosen descriptors
                        corr_test = self.check_if_descriptor_is_correlated_with_chosen_descriptors(
                            descriptor, chosen_descriptors[cluster], correlations_df
                        )
                        if not corr_test[0]:

                            self.logger.info(f"Adding {row['descriptor']} to chosen descriptors")

                            chosen_descriptors[cluster][descriptor] = {
                                "iou": descriptor_avg_iou,
                                "vertex_coverage": row["vertex_coverage"],
                            }

                        else:

                            self.logger.info(
                                f"{row['descriptor']} is correlated with {corr_test[1]} with corr {corr_test[2]}"
                            )

                    else:
                        self.logger.info(
                            f"{row['descriptor']} has high IoU with {iou_test[1]} with iou {iou_test[2]}, skipping"
                        )

            self.logger.info(f"currently chosen descriptors: {chosen_descriptors[cluster]}")
            self.logger.info(f"Finished cluster {cluster}")
            print("*" * 100)

        self.logger.info(
            f"There are {sum([len(descriptors) for descriptors in chosen_descriptors.values()])} chosen descriptors"
        )
        self.logger.info(f"chosen descriptors: {[list(chosen_descriptors[x].keys()) for x in chosen_descriptors]}")
        return chosen_descriptors

    def or_mode(self):
        # get the df that contains the vertex coverage of each descriptor
        summary_df = pd.read_csv(self.working_dir / "vertex_heatmaps" / "summary.csv")

        self.logger.info(f"There are total of {len(summary_df)} descriptors")

        # get the correlation df
        correlations_df = self.get_corr_df(self.images_dir)

        # get the iou df
        ious_df = pd.read_csv(self.working_dir / "vertex_heatmaps" / "ious.csv")

        # get the clusters
        clusters = self.get_clusters()
        chosen_descriptors = {cluster_id: {} for cluster_id in clusters}

        # add column of cluster to summary df
        summary_df["cluster"] = summary_df["descriptor"].apply(lambda x: self.find_cluster_of_descriptor(x, clusters))

        # start iterating over the clusters
        for cluster, cluster_df in summary_df.groupby("cluster"):

            self.logger.info(f"Working on cluster {cluster}")

            # sort the cluster df by vertex coverage
            cluster_df = cluster_df.sort_values(by="vertex_coverage", ascending=False)

            # iterate over the cluster descriptors
            for i, (_, row) in enumerate(cluster_df.iterrows()):

                descriptor = row["descriptor"]
                if descriptor not in correlations_df.columns:
                    continue
                self.logger.info(f"{i} most covering descriptor - {descriptor}")

                # if this is the descriptor with the highest vertex coverage - add it
                if chosen_descriptors[cluster] == {}:

                    self.logger.info(f"Adding {row['descriptor']} to chosen descriptors")

                    descriptor = row["descriptor"]

                    descriptor_avg_iou = self.get_descriptor_iou(descriptor, ious_df)

                    chosen_descriptors[cluster][descriptor] = {
                        "iou": descriptor_avg_iou,
                        "vertex_coverage": row["vertex_coverage"],
                    }

                else:

                    # get descriptor avg iou with all other descriptors in the total dataset
                    descriptor_avg_iou = self.get_descriptor_iou(descriptor, ious_df)

                    # check descriptor's iou with chosen descriptors. if it is above threshold - skip
                    iou_test = self.candidate_iou_with_chosen_descriptors(
                        descriptor, chosen_descriptors[cluster], ious_df
                    )

                    # check if the descriptor is correlated with the chosen descriptors
                    corr_test = self.check_if_descriptor_is_correlated_with_chosen_descriptors(
                        descriptor, chosen_descriptors[cluster], correlations_df
                    )

                    if iou_test[0] or not corr_test[0]:

                        self.logger.info(f"Adding {row['descriptor']} to chosen descriptors")

                        chosen_descriptors[cluster][descriptor] = {
                            "iou": descriptor_avg_iou,
                            "vertex_coverage": row["vertex_coverage"],
                        }

                    else:

                        if iou_test[0]:
                            self.logger.info(
                                f"{row['descriptor']} has high IoU with {iou_test[1]} with iou {iou_test[2]}, skipping"
                            )
                        else:
                            self.logger.info(
                                f"{row['descriptor']} is correlated with {corr_test[1]} with corr {corr_test[2]}"
                            )

            self.logger.info(f"currently chosen descriptors: {chosen_descriptors[cluster]}")
            self.logger.info(f"Finished cluster {cluster}")
            print("*" * 100)

        self.logger.info(
            f"There are {sum([len(descriptors) for descriptors in chosen_descriptors.values()])} chosen descriptors"
        )
        self.logger.info(f"chosen descriptors: {[list(chosen_descriptors[x].keys()) for x in chosen_descriptors]}")
        return chosen_descriptors

    def choose(self):
        if self.mode == "and":
            chosen_descriptors = self.and_mode()
        elif self.mode == "or":
            chosen_descriptors = self.or_mode()

        number_of_descriptors = self.get_num_of_chosen_descriptors(chosen_descriptors)
        if number_of_descriptors > self.max_num_of_descriptors:
            self.logger.info(f"Too many descriptors ({number_of_descriptors}) - reducing")
            while number_of_descriptors > self.max_num_of_descriptors:
                chosen_descriptors = self.reduce_descriptor(chosen_descriptors)
                number_of_descriptors = self.get_num_of_chosen_descriptors(chosen_descriptors)
        elif number_of_descriptors < self.min_num_of_descriptors:
            pass

        with open(self.working_dir / f"chosen_descriptors.json", "w") as f:
            json.dump(chosen_descriptors, f)

        return chosen_descriptors


@hydra.main(config_path="../../config", config_name="choose_algorithm_arik")
def main(cfg: DictConfig) -> None:
    choosing_descriptors = ChoosingDescriptorsArik(cfg)
    choosing_descriptors.choose()


if __name__ == "__main__":
    main()
