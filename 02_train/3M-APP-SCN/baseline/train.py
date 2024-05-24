from __future__ import print_function
from gunpowder import *
from gunpowder.tensorflow import *
import glob
import json
import logging
import math
import numpy as np
import os
import sys
import tensorflow as tf

logging.basicConfig(level=logging.INFO)

# data_dir = "/data/lsd_nm_experiments/03_salk/salk/3M-APP-SCN/training"

data_dir = "/data/lsd_nm_experiments/03_salk/training"

samples = glob.glob(os.path.join(data_dir, "*.zarr"))

neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]


# compute max padding if using elastic augmentation with 45 degree rotations
def calc_max_padding(
    output_size, voxel_size, neighborhood=None, sigma=None, mode="shrink"
):
    if neighborhood is not None:
        if len(neighborhood) > 3:
            neighborhood = neighborhood[9:12]

        max_affinity = Coordinate(
            [np.abs(aff) for val in neighborhood for aff in val if aff != 0]
        )

        method_padding = voxel_size * max_affinity

    if sigma:
        method_padding = Coordinate((sigma * 3,) * 3)

    diag = np.sqrt(output_size[1] ** 2 + output_size[2] ** 2)

    max_padding = Roi(
        (Coordinate([i / 2 for i in [output_size[0], diag, diag]]) + method_padding),
        (0,) * 3,
    ).snap_to_grid(voxel_size, mode=mode)

    return max_padding.get_begin()


def train_until(max_iteration):
    if tf.train.latest_checkpoint("."):
        trained_until = int(tf.train.latest_checkpoint(".").split("_")[-1])
    else:
        trained_until = 0
    if trained_until >= max_iteration:
        return

    with open("train_net.json", "r") as f:
        config = json.load(f)

    raw = ArrayKey("RAW")
    labels = ArrayKey("GT_LABELS")
    labels_mask = ArrayKey("GT_LABELS_MASK")
    artifacts = ArrayKey("ARTIFACTS")
    artifacts_mask = ArrayKey("ARTIFACTS_MASK")
    affs = ArrayKey("PREDICTED_AFFS")
    gt = ArrayKey("GT_AFFINITIES")
    gt_mask = ArrayKey("GT_AFFINITIES_MASK")
    gt_scale = ArrayKey("GT_AFFINITIES_SCALE")
    affs_gradient = ArrayKey("AFFS_GRADIENT")

    # voxel_size = Coordinate((20, 9, 9))
    voxel_size = Coordinate((50, 10, 10))
    input_size = Coordinate(config["input_shape"]) * voxel_size
    output_size = Coordinate(config["output_shape"]) * voxel_size

    # max labels padding calculated
    # labels_padding = Coordinate((500, 369, 369)) => 25,41,41 => 50,82,82
    # labels_padding = Coordinate((1000, 400, 400))
    labels_padding = (input_size - output_size) / 2
    print('labels_padding', labels_padding)
    # labels_padding = calc_max_padding(output_size, voxel_size, sigma=100)

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(gt, output_size)
    request.add(gt_mask, output_size)
    request.add(gt_scale, output_size)

    snapshot_request = BatchRequest({affs: request[gt], affs_gradient: request[gt]})

    data_sources = tuple(
        ZarrSource(
            sample,
            datasets={
                raw: "raw",
                labels: "labels",
                labels_mask: "labels_mask",
            },
            array_specs={
                raw: ArraySpec(interpolatable=True),
                labels: ArraySpec(interpolatable=False),
                labels_mask: ArraySpec(interpolatable=False),
            },
        )
        + Normalize(raw)
        + Pad(raw, None)
        + Pad(labels, labels_padding)
        + Pad(labels_mask, labels_padding)
        + RandomLocation(min_masked=0.5, mask=labels_mask)
        for sample in samples
    )

    train_pipeline = (
        data_sources
        + RandomProvider()
        + ElasticAugment(
            control_point_spacing=[4, 4, 10],
            jitter_sigma=[0, 2, 2],
            rotation_interval=[0, math.pi / 2.0],
            prob_slip=0.05,
            prob_shift=0.05,
            max_misalign=10,
            subsample=8,
        )
        + SimpleAugment(transpose_only=[1, 2])
        + IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1, z_section_wise=True)
        + GrowBoundary(labels, labels_mask, steps=1, only_xy=True)
        + AddAffinities(
            neighborhood,
            labels=labels,
            affinities=gt,
            labels_mask=labels_mask,
            affinities_mask=gt_mask,
        )
        + BalanceLabels(gt, gt_scale, gt_mask)
        + IntensityScaleShift(raw, 2, -1)
        + PreCache(cache_size=40, num_workers=10)
        + Train(
            "train_net",
            optimizer=config["optimizer"],
            loss=config["loss"],
            inputs={
                config["raw"]: raw,
                config["gt_affs"]: gt,
                config["affs_loss_weights"]: gt_scale,
            },
            outputs={config["affs"]: affs},
            gradients={config["affs"]: affs_gradient},
            summary=config["summary"],
            log_dir="log",
            save_every=10000,
        )
        + IntensityScaleShift(raw, 0.5, 0.5)
        + Snapshot(
            {
                raw: "raw",
                labels: "labels",
                gt: "gt_affs",
                affs: "pred_affs",
                gt_mask: "gt_mask",
                labels_mask: "labels_mask",
                affs_gradient: "affs_gradient",
            },
            dataset_dtypes={labels: np.uint16},
            every=5000,
            output_filename="batch_{iteration}.hdf",
            additional_request=snapshot_request,
        )
        + PrintProfilingStats(every=10)
    )

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(max_iteration - trained_until):
            b.request_batch(request)
    print("Training finished")


if __name__ == "__main__":
    iteration = int(sys.argv[1])
    train_until(iteration)
