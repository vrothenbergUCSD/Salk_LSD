from funlib.learn.torch.models import UNet, ConvPass
from lsd.train.gp import AddLocalShapeDescriptor
import gunpowder as gp
import logging
import math
import numpy as np
import torch
import sys

logging.basicConfig(level=logging.INFO)

# helps speed up training
torch.backends.cudnn.benchmark = True

data_path = "/data/lsd_nm_experiments/03_salk/salk/3M-APP-SCN/training/data.zarr"

# compute max padding if using elastic augmentation with 45 degree rotations
def calc_max_padding(
    output_size, voxel_size, neighborhood=None, sigma=None, mode="shrink"
):
    if neighborhood is not None:
        if len(neighborhood) > 3:
            neighborhood = neighborhood[9:12]

        max_affinity = gp.Coordinate(
            [np.abs(aff) for val in neighborhood for aff in val if aff != 0]
        )

        method_padding = voxel_size * max_affinity

    if sigma:
        method_padding = gp.Coordinate((sigma * 3,) * 3)

    diag = np.sqrt(output_size[1] ** 2 + output_size[2] ** 2)

    max_padding = gp.Roi(
        (gp.Coordinate([i / 2 for i in [output_size[0], diag, diag]]) + method_padding),
        (0,) * 3,
    ).snap_to_grid(voxel_size, mode=mode)

    return max_padding.get_begin()


# creates a weighted mse loss with combined lsd and affs losses
class WeightedMSELoss(torch.nn.MSELoss):
    def __init__(self) -> None:
        super(WeightedMSELoss, self).__init__()

    def _calc_loss(self, prediction, target, weights):
        scaled = weights * (prediction - target) ** 2

        if len(torch.nonzero(scaled)) != 0:
            mask = torch.masked_select(scaled, torch.gt(weights, 0))
            loss = torch.mean(mask)

        else:
            loss = torch.mean(scaled)

        return loss

    def forward(
        self,
        pred_lsds=None,
        gt_lsds=None,
        lsds_weights=None,
        pred_affs=None,
        gt_affs=None,
        affs_weights=None,
    ):
        lsd_loss = self._calc_loss(pred_lsds, gt_lsds, lsds_weights)
        aff_loss = self._calc_loss(pred_affs, gt_affs, affs_weights)

        return lsd_loss + aff_loss


# creates a multitask lsd network with simple output heads
class MTLSDModel(torch.nn.Module):
    def __init__(self, unet, num_fmaps):
        super(MTLSDModel, self).__init__()

        self.unet = unet
        self.aff_head = ConvPass(num_fmaps, 3, [[1, 1, 1]], activation="Sigmoid")
        self.lsd_head = ConvPass(num_fmaps, 10, [[1, 1, 1]], activation="Sigmoid")

    def forward(self, input):
        x = self.unet(input)
        lsds = self.lsd_head(x)
        affs = self.aff_head(x)
        return lsds, affs


def pipeline(iterations):
    # create our array keys
    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    labels_mask = gp.ArrayKey("LABELS_MASK")
    gt_affs = gp.ArrayKey("GT_AFFS")
    gt_lsds = gp.ArrayKey("GT_LSDS")
    affs_weights = gp.ArrayKey("AFFS_WEIGHTS")
    gt_affs_mask = gp.ArrayKey("AFFS_MASK")
    pred_affs = gp.ArrayKey("PRED_AFFS")
    pred_lsds = gp.ArrayKey("PRED_LSDS")

    voxel_size = gp.Coordinate((50, 10, 10))

    in_channels = 1
    num_fmaps = 12
    fmap_inc_factor = 5

    # change downsample factors to account for translation equivariant network.
    # If using 3 rather than 2, you need to increase the input shape of the
    # network since more data is shaved off
    downsample_factors = [(1, 2, 2), (1, 2, 2), (2, 2, 2)]

    # create unet. lsd feats = 10, affs feats = 3, but output of network will be
    # 12 to match the input features. so set num_fmaps_out to 14 to account.
    unet = UNet(
        in_channels,
        num_fmaps,
        fmap_inc_factor,
        downsample_factors,
        constant_upsample=True,
        num_fmaps_out=14,
    )

    model = MTLSDModel(unet, num_fmaps=14)

    loss = WeightedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-4, betas=(0.95, 0.999))

    # we can get the output shape by passing a dummy tensor through our network,
    # so we don't have to manually calculate it
    input_shape = [84, 148, 148]
    output_shape = model.forward(torch.empty(size=[1, 1] + input_shape))[0].shape[2:]

    # get to world units
    input_size = gp.Coordinate(input_shape) * voxel_size
    output_size = gp.Coordinate(output_shape) * voxel_size

    # for our lsds, i think setting to 120 triggers an assertion error in the
    # lsd node, which i thought was fixed. anyway, 100 seems to work and
    # probably better anyway (in general like to use around 10 pixels * voxel size)
    sigma = 100

    # compute labels padding needed for extreme elastic augments
    labels_padding = calc_max_padding(output_size, voxel_size, sigma=sigma)

    request = gp.BatchRequest()

    # add keys to batch request
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(gt_affs, output_size)
    request.add(gt_lsds, output_size)
    request.add(affs_weights, output_size)
    request.add(gt_affs_mask, output_size)
    request.add(pred_affs, output_size)
    request.add(pred_lsds, output_size)

    # get our data
    source = gp.ZarrSource(
        # "../data.zarr",
        data_path,
        {
            raw: "volumes/raw",
            labels: "volumes/labels/neuron_ids",
            labels_mask: "volumes/labels/labels_mask",
        },
        {
            raw: gp.ArraySpec(interpolatable=True),
            labels: gp.ArraySpec(interpolatable=False),
            labels_mask: gp.ArraySpec(interpolatable=False),
        },
    )

    # normalize, pad, get random location
    source += gp.Normalize(raw)
    source += gp.Pad(raw, None)
    source += gp.Pad(labels, labels_padding)
    source += gp.Pad(labels_mask, labels_padding)
    source += gp.RandomLocation()

    pipeline = source

    # don't really need this since there is a single upstream source, but
    # keeping it here as it is useful if you end up training on mutiple sources
    pipeline += gp.RandomProvider()

    # adds elastic augment
    pipeline += gp.ElasticAugment(
        control_point_spacing=[5, 5, 10],
        jitter_sigma=[0, 2, 2],
        rotation_interval=[0, math.pi / 2.0],
        prob_slip=0.05,  # good if you have data slips/shifts
        prob_shift=0.05,  # good if you have data slips/shifts
        max_misalign=10,  # good if you have data slips/shifts
        subsample=8,  # for computational efficiency
    )

    # adds random rotations/transposes (in xy since anisotropic)
    pipeline += gp.SimpleAugment(transpose_only=[1, 2])

    # randomly adjust intensity
    pipeline += gp.IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1)

    # grow boundary, honeslty might be fine to get rid of since you have
    # background anyway
    pipeline += gp.GrowBoundary(labels, labels_mask, steps=1, only_xy=True)

    # adds descriptor, we don't need an lsds mask since we use labels_mask as
    # weights for the network (in order to predict zeros in background)
    pipeline += AddLocalShapeDescriptor(
        labels,
        gt_lsds,
        sigma=sigma,
        downsample=2,
    )

    # adds affinities, can still use affs mask for weights, they should handle
    # this intrinsically
    pipeline += gp.AddAffinities(
        affinity_neighborhood=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
        labels=labels,
        affinities=gt_affs,
        labels_mask=labels_mask,
        affinities_mask=gt_affs_mask,
        dtype=np.float32,
    )

    # create weights after factoring in masks
    pipeline += gp.BalanceLabels(gt_affs, affs_weights, mask=gt_affs_mask)

    # pytorch expects shape (b,c,z,y,x). We currently have z,y,x, so add channel
    # dim and then stack batch size (just one for 3d to not be computationally
    # inefficient
    pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Stack(1)

    # computational efficiency
    pipeline += gp.PreCache(cache_size=40, num_workers=10)

    # gunpowder wrapper around train logic
    pipeline += gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs={"input": raw},
        loss_inputs={
            0: pred_lsds,
            1: gt_lsds,
            2: labels_mask,
            3: pred_affs,
            4: gt_affs,
            5: affs_weights,
        },
        outputs={0: pred_lsds, 1: pred_affs},
        save_every=10000,
        log_dir="log" # if you want to log to tensorboard
    )

    # remove batch, channel dims where necessary for viewing
    pipeline += gp.Squeeze([raw, gt_lsds, pred_lsds, gt_affs, pred_affs])
    pipeline += gp.Squeeze([raw])

    # save batch for viewing
    pipeline += gp.Snapshot(
        dataset_names={
            raw: "raw",
            labels: "labels",
            gt_lsds: "gt_lsds",
            pred_lsds: "pred_lsds",
            gt_affs: "gt_affs",
            pred_affs: "pred_affs",
        },
        output_filename="batch_{iteration}.zarr",
        every=10000,
    )

    # construct pipeline, request batches
    with gp.build(pipeline):
        for i in range(iterations):
            pipeline.request_batch(request)

if __name__ == "__main__":
    iteration = int(sys.argv[1])
    if not isinstance(iteration, int):
        iteration = 50000
    pipeline(iteration)
