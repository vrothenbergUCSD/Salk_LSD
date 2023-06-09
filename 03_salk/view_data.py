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
    default="0.0.0.0",
    help="The bind address to use",
)

parser.add_argument(
    "--bind-port",
    "-p",
    type=int,
    default=8080,  # default to 0 (random port)
    help="The bind port to use",
)

args = parser.parse_args()

neuroglancer.set_server_bind_address(args.bind_address, args.bind_port)

f = zarr.open(args.data_dir[0])

datasets = ["volumes/raw", "volumes/labels/neuron_ids", "volumes/labels/labels_mask"]

viewer = neuroglancer.Viewer()

with viewer.txn() as s:
    for ds in datasets:
        res = f[ds].attrs["resolution"]
        offset = f[ds].attrs["offset"]

        dims = neuroglancer.CoordinateSpace(
            names=["z", "y", "x"], units="nm", scales=res
        )

        data = f[ds][:]

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

aws_external = 'ec2-18-236-204-101.us-west-2.compute.amazonaws.com'
aws_internal = 'ip-172-31-1-72.us-west-2.compute.internal'
url = viewer.get_viewer_url()
new_url = url.replace(aws_internal, aws_external)
print(new_url)
