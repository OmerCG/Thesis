from clip2mesh.data_management.data_creation.clip_encoder import generate_clip_scores
from clip2mesh.data_management.data_observation.choosing_descriptors import ChoosingDescriptors


if __name__ == "__main__":
    df_path = "/home/nadav2/dev/data/CLIP2Shape/outs/vertices_heatmap/optimizations/compared_to_inv/flame_shape_multiview_l2/vertex_heatmaps/ious.csv"
    descriptors_groups_json = "/home/nadav2/dev/data/CLIP2Shape/outs/clustering_images/words_jsons/flame_shape.json"
    imgs_dir = "/home/nadav2/dev/data/CLIP2Shape/images/flame_shape_py3d"
    choosing_descriptors = ChoosingDescriptors(verbose=True)
    finalists_descriptors, num_of_descriptors = choosing_descriptors.choose(
        df_path, descriptors_groups_json, max_descriptors_overall=12, min_descriptors_overall=12
    )
    print(finalists_descriptors)

    labels = [[label] for label in finalists_descriptors]
    generate_clip_scores(device="cuda", side=False, imgs_dir=imgs_dir, labels=labels)
