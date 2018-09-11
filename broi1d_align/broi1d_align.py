import torch
from torch import nn

from .crop_and_resize import CropAndResizeFunction


class BRoI1DAlign(nn.Module):

    def __init__(self, crop_depth, temporal_scale, start_pooled_depth, end_pooled_depth, 
                 bratio, extrapolation_value=0, transform_fpcoor=True):
        super(BRoI1DAlign, self).__init__()

        self.crop_depth = crop_depth
        self.temporal_scale = temporal_scale
        self.start_pooled_depth = start_pooled_depth
        self.end_pooled_depth = end_pooled_depth
        self.bratio = bratio
        self.extrapolation_value = extrapolation_value
        self.transform_fpcoor = transform_fpcoor

    def forward(self, featuremap, rois):
        """
        RoIAlign based on crop_and_resize.
        See more details on https://github.com/ppwwyyxx/tensorpack/blob/6d5ba6a970710eaaa14b89d24aace179eb8ee1af/examples/FasterRCNN/model.py#L301
        :param featuremap: NxCxHxW
        :param boxes: Mx4 float box with (x1, y1, x2, y2) **without normalization**
        :param box_ind: M
        :return: MxCxoHxoW
        """
        return CropAndResizeFunction(self.crop_depth, self.temporal_scale, self.start_pooled_depth, 
                                     self.end_pooled_depth, self.bratio, self.extrapolation_value)(featuremap, rois)
