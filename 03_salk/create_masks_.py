import numpy as np
import zarr

container = zarr.open("training/GT_bouton_61.zarr", "a")

labels = container["labels"]
offset = labels.attrs["offset"]
resolution = labels.attrs["resolution"]

labels = labels[:]

# true where labels == true, can be used to get network to make predictions in
# background regions (in this case these regions correspond to glia)
unlabelled = (labels > 0).astype(np.uint8)

# Not needed for GT_bouton_*.zarr
# # label 41 is a big cell body. Let's also create a mask without this label that
# # we can use to restrict random locations to mostly neuropil regions
# object_mask = unlabelled.copy()
# object_mask[labels == 41] = 0

# write data to zarr
for ds_name, data in [
    # ("object_mask", object_mask), 
    ("unlabelled", unlabelled)]:
    container[f"{ds_name}"] = data
    container[f"{ds_name}"].attrs["offset"] = offset
    container[f"{ds_name}"].attrs["resolution"] = resolution
