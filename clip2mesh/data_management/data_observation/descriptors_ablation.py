import hydra
from hydra import compose, initialize
from clip2mesh.data_management.data_creation.clip_encoder import generate_clip_scores
from clip2mesh.data_management.data_observation.choosing_descriptors import ChoosingDescriptors
from clip2mesh.data_management.data_observation.choosing_descriptors_arik import ChoosingDescriptorsArik


if __name__ == "__main__":
    df_path = "/home/nadav2/dev/data/CLIP2Shape/outs/vertices_heatmap/optimizations/compared_to_inv/flame_shape_multiview_l2/vertex_heatmaps/ious.csv"
    descriptors_groups_json = "/home/nadav2/dev/data/CLIP2Shape/outs/clustering_images/words_jsons/flame_shape.json"
    imgs_dir = "/home/nadav2/dev/data/CLIP2Shape/images/smplx_py3d_male"
    arik = True
    if arik:
        with initialize(config_path="../../config"):
            config = compose(config_name="choose_algorithm_arik")
            choosing_descriptors = ChoosingDescriptorsArik(config)
            chosen_descriptors = choosing_descriptors.choose()
            labels = [[x] for x in choosing_descriptors.flatten_dict_of_dicts(chosen_descriptors).keys()]
    else:
        choosing_descriptors = ChoosingDescriptors(verbose=True)
        finalists_descriptors, num_of_descriptors = choosing_descriptors.choose(
            df_path, descriptors_groups_json, max_descriptors_overall=12, min_descriptors_overall=12
        )
        print(finalists_descriptors)

        labels = [[label] for label in finalists_descriptors]
    print(f"stary encoding labels: {labels}")
    generate_clip_scores(device="cuda", side=False, imgs_dir=imgs_dir, labels=labels)
