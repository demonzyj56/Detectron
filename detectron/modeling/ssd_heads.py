"""Single shot multibox detector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from detectron.core.config import cfg
import detectron.core.ssd_factory as ssd_factory
from detectron.ops.multibox_cls_weights import MultiBoxClsWeightOp
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
import detectron.utils.blob as blob_utils


def add_ssd320_extra_layers(model, blob_in, dim_in, dim_mid=128, dim_out=256):
    # 10x10 -> 5x5
    cur = model.Conv(
        blob_in,
        'conv6_1',
        dim_in,
        dim_mid,
        kernel=1,
        stride=1,
        pad=0
    )
    cur = model.Relu(cur, cur)
    cur = model.Conv(
        cur,
        'conv6_2',
        dim_mid,
        dim_out,
        kernel=3,
        stride=2,
        pad=1
    )
    cur = model.Relu(cur, cur)
    # 5x5 -> 3x3
    cur = model.Conv(
        cur,
        'conv7_1',
        dim_out,
        dim_mid,
        kernel=1,
        stride=1,
        pad=0
    )
    cur = model.Relu(cur, cur)
    cur = model.Conv(
        cur,
        'conv7_2',
        dim_mid,
        dim_out,
        kernel=3,
        stride=1,
        pad=0
    )
    cur = model.Relu(cur, cur)
    # 3x3 -> 1x1
    cur = model.Conv(
        cur,
        'conv8_1',
        dim_out,
        dim_mid,
        kernel=1,
        stride=1,
        pad=0
    )
    cur = model.Relu(cur, cur)
    cur = model.Conv(
        cur,
        'conv8_2',
        dim_mid,
        dim_out,
        kernel=3,
        stride=1,
        pad=0
    )
    cur = model.Relu(cur, cur)

    return cur


def add_ssd_outputs(model, blob_in, dim_in):
    level_info = ssd_factory.get_level_info_func()
    _add_ssd_extra_layers(model, blob_in, dim_in)
    cls_output, reg_output, spatial_prob = _add_ssd_decision_layers(
        model, level_info.blobs, level_info.dims, level_info.mbox
    )
    # hard negative mining
    if model.train:
        python_op_name = 'MultiBoxClsWeightOp:' + ','.join(
            ['spatial_prob', 'ssd_labels_int32']
        )
        # Scoped automatically by caffe2.
        model.net.Python(MultiBoxClsWeightOp(
            cfg.SSD.NEG_POS_RATIO,
            cfg.TRAIN.IMS_PER_BATCH
        ).forward)(
            [spatial_prob, 'ssd_labels_int32'],
            ['ssd_label_weights'],
            name=python_op_name
        )


def _add_ssd_extra_layers(model, blob_in, dim_in):
    """Factory method to decide which output layers to use."""
    return globals()['add_ssd{}_extra_layers'.format(cfg.SSD.MIN_DIM)](
        model=model,
        blob_in=blob_in,
        dim_in=dim_in
    )


def _add_ssd_decision_layers(model, blobs_in, dims_in, mbox):
    """Add a convolution layer for each source layer, and then reshape and
    concatenate the outputs to form a 2-D blob of size [N x num_priors, A].
    Output for convolutions are 4-D blobs where the number of channels
    C = num_anchors * num_classes (for classification) or
    C = num_anchors * 4 (for regression).

    Also output the classification probability score which is need for testing
    and hard negative mining.
    """
    cls_outputs, reg_outputs, spatial_probs = [], [], []
    for blob, dim, box in zip(blobs_in, dims_in, mbox):
        cs = model.Conv(
            blob,
            blob + '_cls_score',
            dim,
            box * model.num_classes,
            weight_init=gauss_fill(0.01),
            bias_init=const_fill(0.0)
        )
        rs = model.Conv(
            blob,
            blob + '_bbox_pred',
            dim,
            box * 4,
            weight_init=gauss_fill(0.01),
            bias_init=const_fill(0.0)
        )
        sp = model.net.GroupSpatialSoftmax(
            cs,
            cs + '_spatial_prob',
            num_classes=model.num_classes
        )
        cls_outputs.append(cs)
        reg_outputs.append(rs)
        spatial_probs.append(sp)
    cls_output = _reshape_and_concat_outputs(
        model, cls_outputs, 'ssd_cls_output', model.num_classes
    )
    reg_output = _reshape_and_concat_outputs(
        model, reg_outputs, 'ssd_reg_output', 4
    )
    spatial_prob = _reshape_and_concat_outputs(
        model, spatial_probs, 'ssd_spatial_prob', model.num_classes
    )

    return cls_output, reg_output, spatial_prob


def add_ssd_losses(model):
    """SSD multibox loss.  We put the tensor shape here for reference.
    loss_cls:
        Input:
            ssd_cls_output: (batch_size x num_priors, num_classes)
            ssd_labels_int32: (batch_size x num_priors, )
            ssd_labels_weights: (batch_size x num_priors, )
        Output:
            cls_prob:
            loss_cls:

    loss_bbox:
        Input:
            ssd_reg_output: (batch_size x num_priors, 4)
            ssd_bbox_targets: (batch_size x num_priors, 4)
            ssd_bbox_inside_weights: (batch_size x num_priors, 4)
            ssd_bbox_outside_weights: (batch_size x num_priors, 4)
        Output:
            loss_bbox:
    """
    cls_prob, loss_cls = model.net.SoftmaxWithLoss(
        ['ssd_cls_output', 'ssd_labels_int32', 'ssd_label_weights'],
        ['cls_prob', 'loss_cls'],
        scale=model.GetLossScale()
    )
    loss_bbox = model.net.SmoothL1Loss(
        ['ssd_reg_output', 'ssd_bbox_targets', 'ssd_bbox_inside_weights',
         'ssd_bbox_outside_weights'],
        'loss_bbox',
        scale=model.GetLossScale()
    )
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls, loss_bbox])
    model.Accuracy(['cls_prob', 'ssd_labels_int32'], 'accuracy_cls')
    model.AddLosses(['loss_cls', 'loss_bbox'])
    model.AddMetrics('accuracy_cls')
    return loss_gradients


def _reshape_and_concat_outputs(model, blobs_in, blob_out, dim):
    """For output blobs in multiple layers, do the following:
    1) Transpose from [N, K*A, H, W] to [N, H, W, K*A];
    2) Reshape to [N, num_priors, A];
    3) Concatenate on the second dimension (priors).
    4) Reshape further to [N*num_priors, A]
    Here A is the output dim (num_classes for classification and 4 for
    regression).
    """
    blobs_out = []
    for idx, blob in enumerate(blobs_in):
        blob_tr = model.transpose(blob, blob + '_tr', axes=(0, 2, 3, 1))
        blob_re, _ = model.net.Reshape(
            blob_tr,
            [
                blob_tr,
                blob_tr + '_old_shape_{:d}'.format(idx)
            ],
            shape=(0, -1, dim)
        )
        blobs_out.append(blob_re)
    output = model.concat(blobs_out, blob_out, axis=1)
    output, _ = model.net.Reshape(
        output, [output, output + '_old_shape'], shape=(-1, dim)
    )

    return output

