import glob
import numpy as np
import os
import zarr

volumes = ["3M-APP-SCN", 
        #    "hemi", 
        #    "zebrafinch"
        ]

samples = [glob.glob(os.path.join(f"training", "*.zarr")) for v in volumes]

samples = [i for s in samples for i in s]

labels_name = "labels"
labels_mask_name = "labels_mask"

for sample in samples:
    f = zarr.open(sample, "a")

    labels = f[labels_name][:]
    offset = f[labels_name].attrs["offset"]
    resolution = f[labels_name].attrs["resolution"]

    labels_mask = np.ones_like(labels).astype(np.uint8)


    f[labels_mask_name] = labels_mask
    f[labels_mask_name].attrs["offset"] = offset
    f[labels_mask_name].attrs["resolution"] = resolution
