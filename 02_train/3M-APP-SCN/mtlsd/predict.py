import gunpowder as gp
import logging
import math
import numpy as np
import os
import sys
import torch
import zarr
from funlib.learn.torch.models import UNet, ConvPass

logging.basicConfig(level=logging.INFO)


class MTLSDModel(torch.nn.Module):
    def __init__(self, unet, num_fmaps):
        super(MTLSDModel, self).__init__()

        self.unet = unet
        self.aff_head = ConvPass(num_fmaps, 3, [[1, 1, 1]], activation="Sigmoid")
        self.lsd_head = ConvPass(num_fmaps, 10, [[1, 1, 1]], activation="Sigmoid")

    def forward(self, input):
        x = self.unet(input)

        lsds = self.lsd_head(x[0])  # decoder 1
        affs = self.aff_head(x[1])  # decoder 2
        return lsds, affs


def predict(raw_file, raw_dataset, out_file, out_datasets):
    raw = gp.ArrayKey("RAW")
    pred_affs = gp.ArrayKey("PRED_AFFS")
    pred_lsds = gp.ArrayKey("PRED_LSDS")

    voxel_size = gp.Coordinate((50, 10, 10))

    in_channels = 1
    num_fmaps = 12
    fmap_inc_factor = 5

    downsample_factors = [(1, 2, 2), (1, 2, 2), (2, 2, 2)]

    kernel_size_down = [
        [(3,) * 3, (3,) * 3],
        [(3,) * 3, (3,) * 3],
        [(3,) * 3, (3,) * 3],
        [(1, 3, 3), (1, 3, 3)],
    ]

    kernel_size_up = [
        [(1, 3, 3), (1, 3, 3)],
        [(3,) * 3, (3,) * 3],
        [(3,) * 3, (3,) * 3],
    ]

    unet = UNet(
        in_channels,
        num_fmaps,
        fmap_inc_factor,
        downsample_factors,
        kernel_size_down,
        kernel_size_up,
        constant_upsample=True,
        num_fmaps_out=14,
        num_heads=2,  # uses two decoders
    )

    model = MTLSDModel(unet, num_fmaps=14)

    input_shape = [48, 196, 196]
    output_shape = model.forward(torch.empty(size=[1, 1] + input_shape))[0].shape[2:]

    input_size = gp.Coordinate(input_shape) * voxel_size
    output_size = gp.Coordinate(output_shape) * voxel_size

    context = (input_size - output_size) / 2

    model.eval()

    scan_request = gp.BatchRequest()

    scan_request.add(raw, input_size)
    scan_request.add(pred_affs, output_size)
    scan_request.add(pred_lsds, output_size)

    source = gp.ZarrSource(
        raw_file, {raw: raw_dataset}, {raw: gp.ArraySpec(interpolatable=True)}
    )

    with gp.build(source):
        total_input_roi = source.spec[raw].roi
        total_output_roi = total_input_roi.grow(-context, -context)

    f = zarr.open(out_file, "w")

    for ds_name, channels in out_datasets:
        print(ds_name, channels)

        ds = f.create_dataset(
            ds_name,
            shape=[channels] + list(total_output_roi.get_shape() / voxel_size),
            dtype=np.uint8,
        )

        ds.attrs["resolution"] = voxel_size
        ds.attrs["offset"] = total_output_roi.get_offset()

    predict = gp.torch.Predict(
        model,
        checkpoint=model_checkpoint,
        inputs={"input": raw},
        outputs={
            0: pred_lsds,
            1: pred_affs,
        },
    )

    write = gp.ZarrWrite(
        dataset_names={
            pred_affs: out_datasets[0][0],
            pred_lsds: out_datasets[1][0],
        },
        output_filename=out_file,
    )

    scan = gp.Scan(scan_request)

    pipeline = (
        source
        + gp.Normalize(raw)
        + gp.Unsqueeze([raw])
        + gp.Unsqueeze([raw])
        + predict
        + gp.Squeeze([pred_affs])
        + gp.Squeeze([pred_lsds])
        + gp.Normalize(pred_affs)
        + gp.IntensityScaleShift(pred_affs, 255, 0)
        + gp.IntensityScaleShift(pred_lsds, 255, 0)
        + write
        + scan
    )

    predict_request = gp.BatchRequest()

    predict_request[raw] = total_input_roi
    predict_request[pred_affs] = total_output_roi
    predict_request[pred_lsds] = total_output_roi

    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)


if __name__ == "__main__":
    checkpoint_iteration = int(sys.argv[1])
    model_checkpoint = f"/data/lsd_nm_experiments/02_train/3M-APP-SCN/mtlsd_04/model_checkpoint_{checkpoint_iteration}"
    raw_file = "/data/lsd_nm_experiments/03_salk/salk/3M-APP-SCN/training/data.zarr"
    # raw_file = "/data/lsd_nm_experiments/03_salk/training/data_expanded.zarr"
    raw_dataset = "raw"
    out_file = f"prediction_expanded_{checkpoint_iteration}.zarr"
    out_datasets = [("pred_affs", 3), ("pred_lsds", 10)]

    predict(raw_file, raw_dataset, out_file, out_datasets)
