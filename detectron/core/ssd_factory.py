"""Factory methods for generating SSD level info."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import numpy as np

from detectron.core.config import cfg


SsdLevelInfo = collections.namedtuple(
    'SsdLevelInfo',
    ['min_dim', 'blobs', 'dims', 'spatial_sizes', 'mbox',]
)


RESNET_FINAL_BLOB = 'res5_2_sum'
RESNET_FINAL_DIM = 2048


def get_level_info_func():
    """Level info lives inside this module, thus globals() suffices for lookup.
    """
    def _get_basenet_name():
        """The name of base network deduced from cfg.MODEL.CONV_BODY. Available
        keys are 'vgg16', 'resnet50', 'resnet101', 'resnet50_32x8d',
        'resnet101_32x8d', ...
        """
        conv_body = cfg.MODEL.CONV_BODY.lower()
        basenet_name = None
        for key in ('vgg16', 'resnet50', 'resnet101'):
            if key in conv_body:
                basenet_name = key
                break
        assert basenet_name is not None, \
            'Cannot deduce basenet name from {}'.format(conv_body)
        if 'resnet' in basenet_name and cfg.RESNETS.NUM_GROUPS > 1:
            # ResNext
            basenet_name += '_{}x{}d'.format(
                cfg.RESNETS.NUM_GROUPS, cfg.RESNETS.WIDTH_PER_GROUP
            )
        return basenet_name

    return globals()['ssd{}_{}_level_info'.format(
        cfg.SSD.MIN_DIM, _get_basenet_name()
    )]()


def ssd320_resnet50_level_info():
    assert cfg.SSD.MIN_DIM == 320
    return SsdLevelInfo(
        min_dim=320,
        blobs=('res3_3_sum', 'res4_5_sum', 'res5_2_sum',
               'conv6_2', 'conv7_2', 'conv8_2'),
        dims=(512, 1024, 2048, 256, 256, 256),
        spatial_sizes=(40, 20, 10, 5, 3, 1),
        mbox=(4, 6, 6, 6, 4, 4),
    )


def ssd320_resnet101_level_info():
    assert cfg.SSD.MIN_DIM == 320
    return SsdLevelInfo(
        min_dim=320,
        blobs=('res3_3_sum', 'res4_22_sum', 'res5_2_sum',
               'conv6_2', 'conv7_2', 'conv8_2'),
        dims=(512, 1024, 2048, 256, 256, 256),
        spatial_sizes=(40, 20, 10, 5, 3, 1),
        mbox=(4, 6, 6, 6, 4, 4),
    )
