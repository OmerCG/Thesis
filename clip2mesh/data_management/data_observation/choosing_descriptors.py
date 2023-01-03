import numpy as np
import pandas as pd

df = pd.read_csv(
    "/home/nadav2/dev/data/CLIP2Shape/outs/vertices_heatmap/optimizations/flame_shape_multiview/vertex_heatmaps/vertex_heatmaps.csv"
)
sorted_indices_values = np.vstack([np.array(eval(df["sorted_indices"].values[i])) for i in range(len(df))])
overlap = np.zeros((len(df), len(df)))
for i in range(len(df)):
    for j in range(len(df)):
        overlap[i, j] = (sorted_indices_values[i] == sorted_indices_values[j]).sum()


# get upper triangle of the matrix
upper_traingle = np.tril(overlap, k=-1)

# get the indices of the descriptors that are overlapping with each descriptor
overlapping_descriptors = {}
for i, descriptor in enumerate(df["descriptor"].values):
    overlapping_descriptors[descriptor] = df["descriptor"][np.where(upper_traingle[i] > 10)[0]].values

# if there are any empty lists, remove them
overlapping_descriptors_dict = {k: v.tolist() for k, v in overlapping_descriptors.items() if len(v) > 0}

# create lists of descriptors to keep and remove
descriptors_to_keep = []
descriptors_to_remove = []

# run on each descriptor and its overlapping descriptors
for descriptor, overlapping_descriptors_list in overlapping_descriptors_dict.items():

    print("\ndescriptor:", descriptor, "|", "overlapping_descriptors:", overlapping_descriptors_list)

    # create lists of descriptors to remove
    overlapping_descriptors_to_remove = []

    # get the effect of the descriptor
    descriptors_effect = df[df["descriptor"] == descriptor]["effect"].values[0]

    # run on each overlapping descriptor
    for overlapped_descriptor in overlapping_descriptors_list:

        # get the effect of the overlapping descriptor
        overlapped_descriptors_effect = df[df["descriptor"] == overlapped_descriptor]["effect"].values[0]

        # if the effect of the descriptor is bigger than the effect of the overlapping descriptor, add the overlapping descriptor to the list of descriptors to remove
        if descriptors_effect > overlapped_descriptors_effect:
            print(f"{descriptor} is more general than {overlapped_descriptor}")
            overlapping_descriptors_to_remove.append(overlapped_descriptor)

        # if the effect of the descriptor is smaller than the effect of the overlapping descriptor, add the descriptor to the list of descriptors to remove and stop the loop
        else:
            print(f"{overlapped_descriptor} is more general than {descriptor}")
            descriptors_to_remove.append(descriptor)
            if (
                overlapped_descriptor not in overlapping_descriptors_dict.keys()
                and overlapped_descriptor not in descriptors_to_keep
            ):
                descriptors_to_keep.append(overlapped_descriptor)
            break

    if len(overlapping_descriptors_to_remove) > 0:
        for descriptor_to_remove in overlapping_descriptors_to_remove:
            overlapping_descriptors_dict[descriptor].remove(descriptor_to_remove)

    if len(overlapping_descriptors_dict[descriptor]) == 0:
        descriptors_to_keep.append(descriptor)

for descriptor in descriptors_to_remove:
    overlapping_descriptors_dict.pop(descriptor)

print("\ndescriptors_to_keep", descriptors_to_keep)
overlapping_descriptors_dict
