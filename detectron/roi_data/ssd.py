"""Minibatch construction for SSD."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import math
import numpy as np
import numpy.random as npr
import cv2

from detectron.core.config import cfg
import detectron.core.ssd_factory as ssd_factory
from detectron.modeling.generate_anchors import generate_anchors
import detectron.roi_data.data_utils as data_utils
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils
from detectron.utils.augmentations import Augmentations

logger = logging.getLogger(__name__)


def get_ssd_blob_names(is_training=True):
    # TODO(leoyolo): TBD for testing
    blob_names = []

    if is_training:
        blob_names += [
            # (batch_size x num_priors, )
            'ssd_labels_int32',
            'ssd_label_weights',
            # (batch_size x num_priors, 4)
            'ssd_bbox_targets',
            'ssd_bbox_inside_weights',
            'ssd_bbox_outside_weights'
        ]

    return blob_names


def add_and_proprecess_ssd_blobs(blobs, roidb):
    # Note: since faster-rcnn supports only flipping with varying sizes as
    # data augmentation, we postpone SSD-style data augmentation here.
    data_augmentor = Augmentations(
        cfg.SSD.MIN_DIM,
        cfg.PIXEL_MEANS,
        cfg.SSD.EXPAND_FACTOR,
        cfg.SSD.CROP_FACTOR
    )
    prior_box = PriorBox(level_info=ssd_factory.get_level_info_func()).forward()
    prior_box = np.concatenate(prior_box, axis=0)
    for im_i, entry in enumerate(roidb):
        im = cv2.imread(entry['image'])
        assert im is not None, \
            'Failed to read image {}'.format(entry['image'])
        im = im.astype(np.float32, copy=False)

        gt_inds = np.where(
            (entry['gt_classes'] > 0) & (entry['is_crowd'] == 0)
        )[0]
        gt_boxes = entry['boxes'][gt_inds, :]
        gt_classes = entry['gt_classes'][gt_inds]

        im, gt_boxes, gt_classes = data_augmentor(im, gt_boxes, gt_classes)
        assert len(gt_boxes) > 0, 'No gt_boxes are created'
        assert gt_boxes.shape[0] == len(gt_classes), \
            'Inconsistent gt_boxes and gt_classes'
        ssd_targets = _compute_ssd_targets(prior_box, gt_boxes, gt_classes)
        blobs['data'].append(im)
        for k, v in ssd_targets:
            blobs[k].append(v)

    # concatenate images
    blobs['data'] = np.concatenate(
        [np.expand_dims(blob, axis=0) for blob in blobs['data']],
        axis=0
    )
    blobs['data'] = blobs['data'].transpose((0, 3, 1, 2))

    # concatenate labels into shape (batch_size x num_priors, ...)
    for k, v in blobs.items():
        if isinstance(v, list) and len(v) > 0:
            blobs[k] = np.concatenate(v)

    # I don't know if any invalid operations involve.
    return True


def _compute_ssd_targets(all_anchors, gt_boxes, gt_classes):
    """Get labels and bbox_targets for ALL prior boxes."""
    # Implementation note: for SSD it does not have the concept of "region of
    # interests" (rois), thus we cannot compute rpn_proposals and bbox_targets
    # when loading the roidb.  We have to compute the targets AFTER loading
    # roidb and doing data augmentation.  This is why we don't use
    # functionality at rpn and fast_rcnn for precomputing the bbox_targets.
    #
    # Also, we don't distribute the labels and weights into respective octave.
    #
    # Refer to roi_data.rpn._get_rpn_blobs for more info.

    total_anchors = all_anchors.shape[0]
    straddle_thresh = cfg.TRAIN.RPN_STRADDLE_THRESH
    if straddle_thresh >= 0:
        # Only keep anchors inside the image by a margin of straddle_thresh
        # Set TRAIN.RPN_STRADDLE_THRESH to -1 (or a large value) to keep all
        # anchors
        inds_inside = np.where(
            (all_anchors[:, 0] >= -straddle_thresh) &
            (all_anchors[:, 1] >= -straddle_thresh) &
            (all_anchors[:, 2] < cfg.SSD.MIN_DIM + straddle_thresh) &
            (all_anchors[:, 3] < cfg.SSD.MIN_DIM + straddle_thresh)
        )[0]
        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
    else:
        inds_inside = np.arange(all_anchors.shape[0])
        anchors = all_anchors
    num_inside = len(inds_inside)

    logger.debug('total_anchors: {}'.format(total_anchors))
    logger.debug('inds_inside: {}'.format(num_inside))
    logger.debug('anchors.shape: {}'.format(anchors.shape))

    labels = np.empty((num_inside, ), dtype=np.int32)
    labels.fill(-1)

    # Compute overlaps between the anchors and the gt boxes overlaps
    anchor_by_gt_overlap = box_utils.bbox_overlaps(anchors, gt_boxes)
    # Map from anchor to gt box that has highest overlap
    anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(axis=1)
    # For each anchor, amount of overlap with most overlapping gt box
    anchor_to_gt_max = anchor_by_gt_overlap[np.arange(num_inside),
                                            anchor_to_gt_argmax]

    # assign positive labels with highest overlap > FG_THRESH
    fg_inds = np.where(anchor_to_gt_max > cfg.TRAIN.FG_THRESH)[0]
    labels[fg_inds] = gt_classes[anchor_to_gt_argmax[fg_inds]]
    # assign negative labels with highest overlap between (low, high)
    bg_inds = np.where(
        (anchor_to_gt_max < cfg.TRAIN.BG_THRESH_HI) &
        (anchor_to_gt_max >= cfg.TRAIN.BG_THRESH_LO)
    )[0]
    labels[bg_inds] = 0

    # reuse the same fg_inds as positive positions
    bbox_targets = np.zeros((num_inside, 4), dtype=np.float32)
    bbox_targets[fg_inds, :] = data_utils.compute_targets(
        anchors[fg_inds, :], gt_boxes[anchor_to_gt_max[fg_inds], :]
    )

    bbox_inside_weights = np.zeros((num_inside, 4), dtype=np.float32)
    bbox_inside_weights[labels > 0, :] = (1.0, 1.0, 1.0, 1.0)

    # Notice that we normalize the loss with all training samples within
    # one minibatch, thus we do not specify the outside_weights, but
    # postpone until the definition of the final multibox_loss.
    bbox_outside_weights = np.zeros((num_inside, 4), dtype=np.float32)
    bbox_outside_weights[labels > 0, :] = 1.0

    # Map up to original set of anchors
    labels = data_utils.unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = data_utils.unmap(
        bbox_targets, total_anchors, inds_inside, fill=0
    )
    bbox_inside_weights = data_utils.unmap(
        bbox_inside_weights, total_anchors, inds_inside, fill=0
    )
    bbox_outside_weights = data_utils.unmap(
        bbox_outside_weights, total_anchors, inds_inside, fill=0
    )

    return dict(
        ssd_labels_int32=labels,
        ssd_bbox_targets=bbox_targets,
        ssd_bbox_inside_weights=bbox_inside_weights,
        ssd_bbox_outside_weights=bbox_outside_weights
    )


################################################################################
# PriorBox
################################################################################

class PriorBox(object):
    """Prior box for SSD at each level."""
    _aspect_ratio_dispatch = {
        4: (0.5, 1., 2.),
        6: (1./3., 0.5, 1., 2., 3.)
    }

    def __init__(self, level_info):
        self._level_info = level_info
        self._min_sizes, self._max_sizes = _get_min_max_sizes(
            level_info.min_dim,
            len(level_info.mbox),
            cfg.SSD.MIN_RATIO,
            cfg.SSD.MAX_RATIO
        )

    def forward(self):
        prior_box = []
        for size, num_box, min_size, max_size in zip(
            self._level_info.spatial_sizes,
            self._level_info.mbox,
            self._min_sizes,
            self._max_sizes
        ):
            stride = self._level_info.min_dim / size
            aspect_ratios = self._aspect_ratio_dispatch[num_box]
            field_of_anchors = _get_default_anchors_single_scale(
                size,
                size,
                min_size,
                max_size,
                aspect_ratios,
                stride
            )
            prior_box.append(field_of_anchors)

        return prior_box


def _get_default_anchors_single_scale(
    feature_height,
    feature_width,
    size_this_scale,
    size_next_scale,
    aspect_ratios,
    stride
):
    """Rewrite data_utils.get_field_of_anchors since we don't need thread
    caching.
    Notice that for SSD the default setting has 4/6 anchors per level/octave:
    aspect ratio (1, 0.5, 2) for original scale + aspect ratio 1 for a slightly
    larger scale (geometric mean of max_size and min_size), or
    aspect ratio (1, 0.5, 2, 1/3, 3) for original scale + aspect ratio 1 for
    the larger scale.
    We respect this setting, but also keep in mind that other more uniform
    settings (as those used in RetinaNet) are also available.
    Anchor coordinates are absolute with respect to size and stride.
    feature_height and feature_width control how many anchors to generate.
    """
    # Generate anchors at this scale: 3/5 depending on aspect ratios
    cell_anchors_this_scale = generate_anchors(
        stride=stride, sizes=(size_this_scale, ), aspect_ratios=aspect_ratios
    )
    # Generate THE anchor between this scale and next scale:
    cell_anchors_between = generate_anchors(
        stride=stride,
        sizes=(np.sqrt(size_this_scale*size_next_scale), ),
        aspect_ratios=(1, )
    )
    cell_anchors = np.concatenate(
        (cell_anchors_this_scale, cell_anchors_between),
        axis=0
    )

    # Refer to data_utils.get_field_of_anchors
    shift_x = np.arange(0, feature_width) * stride
    shift_y = np.arange(0, feature_height) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift_x, shift_y = shift_x.ravel(), shift_y.ravel()
    shifts = np.vstack((shift_x, shift_y, shift_x, shift_y)).transpose()

    A = cell_anchors.shape[0]
    K = shifts.shape[0]
    field_of_anchors = (
        cell_anchors.reshape((1, A, 4)) +
        shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    )
    field_of_anchors = field_of_anchors.reshape((K * A, 4))

    return field_of_anchors


def _get_min_max_sizes(min_dim, num_layers, min_ratio=.2, max_ratio=.9):
    """SSD style determining min and max sizes.
    return:
        min_sizes, max_sizes
    """
    min_sizes, max_sizes = [], []
    # in percentage
    min_ratio = min_ratio * 100
    max_ratio = max_ratio * 100
    step = int(math.floor((max_ratio - min_ratio) / (num_layers - 2)))
    for ratio in range(min_ratio, max_ratio+1, step):
        min_sizes.append(min_dim * ratio / 100)
        max_sizes.append(min_dim * ratio / 100)
    min_sizes = [min_dim * min_ratio / 2 / 100] + min_sizes
    max_sizes = [min_dim * min_ratio / 100] + max_sizes

    return min_sizes, max_sizes

