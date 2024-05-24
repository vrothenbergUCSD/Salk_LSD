import argparse
import neuroglancer
import numpy as np
import os
import sys
import zarr

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data-dir",
    "-d",
    type=str,
    action="append",
    help="The path to the zarr container to show",
)

parser.add_argument(
    "--bind-address",
    "-b",
    type=str,
    default="localhost",
    help="The bind address to use",
)

parser.add_argument(
    "--bind-port",
    "-p",
    type=int,
    default=0,  # default to 0 (random port)
    help="The bind port to use",
)

args = parser.parse_args()

neuroglancer.set_server_bind_address(args.bind_address, args.bind_port)


f = zarr.open(args.data_dir[0])

print(f)

# Add 'pred_affs' to the datasets to visualize
datasets = ["volumes/raw", 
            # "volumes/affs_gradient",
            # "volumes/gt_affinities",
            "volumes/pred_affs",
            "volumes/labels/neuron_ids", 
            # "volumes/labels/gt_mask",
            "volumes/labels/labels_mask", 
            ]

viewer = neuroglancer.Viewer()

with viewer.txn() as s:
    for ds in datasets:
        print('ds:', ds)
        res = f[ds].attrs["resolution"]
        print('res', res)
        if len(res) == 3:
            res = [1] + res
            print('new res:', res)
        offset = list(f[ds].attrs["offset"])
        print('offset:', offset)
        if len(offset) == 3:
            offset = [0] + offset
            print('new offset:', offset)
        offset = [x / y for x, y in zip(offset, res)]
        print('adjusted_offset', offset)

        dims = neuroglancer.CoordinateSpace(
            names=["c^", "z", "y", "x"], units="nm", scales=res
        )
        
        data = f[ds][:]
        print('data:', data.shape)

        # Add additional dimension for 3d raw volumes
        if data.ndim == 3:
            data = data[np.newaxis, :]

        print('data:', data.shape)
        print('dims:', dims)

        # Find the shape of the larger volume 'volumes/raw'
        if ds == 'volumes/raw':
            large_shape = data.shape

        # # If the current volume is 'pred_affs', pad it to match the shape of 'volumes/raw'
        # elif ds == 'pred_affs':
        #     padding = [(0, 0) if i == 0 else (0, large_shape[i] - s) for i, s in enumerate(data.shape)]
        #     data = np.pad(data, padding, mode='constant')

        if "mask" in ds:
            data *= 255

        layer = neuroglancer.LocalVolume(
            data=data, voxel_offset=offset, dimensions=dims
        )

        layer_type = (
            neuroglancer.SegmentationLayer
            if data.dtype == np.uint64
            else neuroglancer.ImageLayer
        )

        s.layers[ds] = layer_type(source=layer)

# aws_address = 'ec2-54-185-85-99.us-west-2.compute.amazonaws.com'
url = viewer.get_viewer_url()
# new_url = url.replace('ip-172-31-1-72.us-west-2.compute.internal', aws_address)
print(url)
