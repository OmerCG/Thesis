import json
import logging
import pandas as pd
from typing import Tuple, Dict, Union


class ChoosingDescriptors:
    def __init__(self, verbose: bool = False):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s: - %(message)s")
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose

    def choose_between_2_descriptors(
        self, df: pd.DataFrame, first_descriptor: str, second_descriptor: str
    ) -> Tuple[str, float]:
        first_descriptor_avg_iou = df[
            (df["descriptor_1"] == first_descriptor) | (df["descriptor_2"] == first_descriptor)
        ]["iou"].mean()
        second_descriptor_avg_iou = df[
            (df["descriptor_1"] == second_descriptor) | (df["descriptor_2"] == second_descriptor)
        ]["iou"].mean()
        if self.verbose:
            self.logger.info(f"{first_descriptor} iou: {first_descriptor_avg_iou}")
        if self.verbose:
            self.logger.info(f"{second_descriptor} iou: {second_descriptor_avg_iou}")
        if first_descriptor_avg_iou < second_descriptor_avg_iou:
            if self.verbose:
                self.logger.info(f"chose {first_descriptor} with iou {first_descriptor_avg_iou}")
            return first_descriptor, first_descriptor_avg_iou
        else:
            if self.verbose:
                self.logger.info(f"chose {second_descriptor} with iou {second_descriptor_avg_iou}")
            return second_descriptor, second_descriptor_avg_iou

    @staticmethod
    def flatten_dict_of_dicts(dict_of_dicts: Dict[int, Dict[str, float]]) -> Dict[str, float]:
        flattened_dict = {}
        for value in dict_of_dicts.values():
            flattened_dict.update(value)
        return flattened_dict

    @staticmethod
    def get_number_of_unique_descriptors(df: Union[pd.DataFrame, str]) -> int:
        if isinstance(df, str):
            df = pd.read_csv(df)
        unique_descriptors = set(df["descriptor_1"].unique().tolist() + df["descriptor_2"].unique().tolist())
        return len(unique_descriptors)

    def initial_filter(
        self, df_path: str, descriptors_groups_json: str, max_descriptors_per_cluster: int = 6
    ) -> Dict[int, Dict[str, float]]:
        df = pd.read_csv(df_path)
        descriptors_groups = json.load(open(descriptors_groups_json, "r"))
        if self.verbose:
            self.logger.info(f"Total number of descriptors to choose from: {self.get_number_of_unique_descriptors(df)}")
            self.logger.info(f"Total number of clusters: {len(descriptors_groups)}")
        finalists_descriptors = {}
        for iou_group, descriptors in descriptors_groups.items():
            if len(descriptors) == 1:
                descriptor_iou = df[
                    (df["descriptor_1"] == second_descriptor) | (df["descriptor_2"] == second_descriptor)
                ]["iou"].mean()
                finalists_descriptors[iou_group] = {descriptors[0]: descriptor_iou}
            else:
                group_df = df[df["descriptor_1"].isin(descriptors) & df["descriptor_2"].isin(descriptors)]
                group_df = group_df.sort_values("iou", ascending=False)
                group_df["group"] = group_df["iou"].apply(lambda x: 1 if x > 0.7 else 0)
                final_candidates = {}
                for group_idx in group_df["group"].unique():
                    chosed_descriptors = {}
                    for _, row in group_df[group_df["group"] == group_idx].iterrows():
                        if self.verbose:
                            print(f"*" * 50)
                        if self.verbose:
                            self.logger.info(f"choosing between: {row['descriptor_1']} | {row['descriptor_2']}")
                        first_descriptor = row["descriptor_1"]
                        second_descriptor = row["descriptor_2"]

                        chosen_descriptor, chosen_descriptor_iou = self.choose_between_2_descriptors(
                            group_df, first_descriptor, second_descriptor
                        )
                        if chosen_descriptor not in chosed_descriptors.keys():
                            if chosed_descriptors == {}:
                                if self.verbose:
                                    self.logger.info(f"first descriptor chosen -> {chosen_descriptor}")
                                chosed_descriptors[chosen_descriptor] = chosen_descriptor_iou
                            else:
                                if self.verbose:
                                    self.logger.info(f"iterating over chosed descriptors -> {chosed_descriptors}")
                                add_descriptor = False
                                for descriptor in chosed_descriptors.keys():
                                    if self.verbose:
                                        self.logger.info(f"choosing between: {descriptor} | {chosen_descriptor}")
                                    sub_chosen_descriptor, _ = self.choose_between_2_descriptors(
                                        group_df, descriptor, chosen_descriptor
                                    )
                                    if sub_chosen_descriptor != chosen_descriptor:
                                        if self.verbose:
                                            self.logger.info(
                                                f"{chosen_descriptor} has higher iou than {descriptor}, hence {descriptor} is chosen"
                                            )
                                        break
                                    else:
                                        add_descriptor = True
                                if add_descriptor:
                                    if self.verbose:
                                        self.logger.info(
                                            f"{chosen_descriptor} has lower iou than all chosed descriptors, hence {chosen_descriptor} is chosen"
                                        )
                                    chosed_descriptors[chosen_descriptor] = chosen_descriptor_iou

                        if self.verbose:
                            self.logger.info(chosed_descriptors)

                    # sorted_chosed_descriptors = sorted(chosed_descriptors, key=chosed_descriptors.get, reverse=False)
                    for descriptor, iou in chosed_descriptors.items():
                        if descriptor not in final_candidates.keys():
                            final_candidates[descriptor] = iou

                if self.verbose:
                    print(f"*" * 50)
                possible_options = df["descriptor_1"].unique().tolist()
                possible_options = possible_options + [
                    item for item in df["descriptor_2"].unique() if item not in possible_options
                ]
                if self.verbose:
                    self.logger.info(f"possible_options: {possible_options} | total {len(possible_options)}")
                if self.verbose:
                    self.logger.info(f"final candidates: {final_candidates} | total {len(final_candidates)}")
                sorted_descriptors_by_iou_ascending = sorted(final_candidates, key=final_candidates.get, reverse=False)[
                    :max_descriptors_per_cluster
                ]
                finalists_descriptors[iou_group] = {
                    descriptor_name: final_candidates[descriptor_name]
                    for descriptor_name in sorted_descriptors_by_iou_ascending
                }

        return finalists_descriptors

    def reduce_descriptor(self, dict_of_desctiptors: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        flattened_dict = self.flatten_dict_of_dicts(dict_of_desctiptors)
        sorted_dict = sorted(flattened_dict, key=flattened_dict.get, reverse=True)
        clusters_counter = {clutser: len(descriptors) for clutser, descriptors in dict_of_desctiptors.items()}
        remove_idx = 0
        while True:
            max_iou_descriptor = sorted_dict[remove_idx]
            cluster_of_descriptor = self.find_cluster_of_descriptor(max_iou_descriptor, dict_of_desctiptors)
            all_clusters_have_one_descriptor = all(
                [cluster_counter == 1 for cluster_counter in clusters_counter.values()]
            )
            if all_clusters_have_one_descriptor:
                if self.verbose:
                    self.logger.info(f"all clusters have one descriptor, that is the minimal number of descriptors")
                return dict_of_desctiptors
            if clusters_counter[cluster_of_descriptor] > 1:
                break
            else:
                remove_idx += 1
        if self.verbose:
            print(f"removing {max_iou_descriptor} from cluster {cluster_of_descriptor}")
        del dict_of_desctiptors[cluster_of_descriptor][max_iou_descriptor]
        return dict_of_desctiptors

    @staticmethod
    def find_cluster_of_descriptor(descriptor: str, dict_of_desctiptors: Dict[str, Dict[str, float]]) -> int:
        for cluster, descriptors_dict in dict_of_desctiptors.items():
            if descriptor in descriptors_dict.keys():
                return cluster

    def choose(
        self,
        df_path: str,
        descriptors_groups_json: str,
        max_descriptors_per_cluster: int = 6,
        max_descriptors_overall: int = 12,
    ):
        initial_filter = self.initial_filter(df_path, descriptors_groups_json, max_descriptors_per_cluster)
        num_of_descriptors = 0
        for descriptors in initial_filter.values():
            num_of_descriptors += len(descriptors)
        if self.verbose:
            print()
            total_number_of_descriptors = self.get_number_of_unique_descriptors(df_path)
            self.logger.info(f"There are {len(initial_filter.keys())} clusters of descriptors")
            self.logger.info(
                f"After filtering, there are {num_of_descriptors} descriptors left out of {total_number_of_descriptors} descriptors"
            )
        if num_of_descriptors > max_descriptors_overall:
            if self.verbose:
                self.logger.info(f"Too many descriptors, choosing only {max_descriptors_overall} descriptors")
            while num_of_descriptors > max_descriptors_overall:
                initial_filter = self.reduce_descriptor(initial_filter)
                num_of_descriptors -= 1

        return initial_filter


df_path = "/home/nadav2/dev/data/CLIP2Shape/outs/vertices_heatmap/optimizations/compared_to_inv/smplx_female_multiview_diff_coord/vertex_heatmaps/ious.csv"
descriptors_groups_json = "/home/nadav2/dev/data/CLIP2Shape/outs/clustering_images/words_jsons/smplx_female.json"


if __name__ == "__main__":
    choosing_descriptors = ChoosingDescriptors(verbose=True)
    finalists_descriptors = choosing_descriptors.choose(df_path, descriptors_groups_json, max_descriptors_overall=12)
    print(finalists_descriptors)
