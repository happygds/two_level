import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from ._ext import crop_and_resize as _backend


class CropAndResizeFunction(Function):

    def __init__(self, crop_depth, temporal_scale, start_pooled_depth, 
                 end_pooled_depth, bratio, extrapolation_value=0):
        self.crop_depth = crop_depth
        self.temporal_scale = temporal_scale
        self.start_pooled_depth = start_pooled_depth
        self.end_pooled_depth = end_pooled_depth
        self.bratio = bratio
        self.extrapolation_value = extrapolation_value

    def forward(self, features, rois):
        batch_size, num_channels, data_depth = features.size()
        num_rois = rois.size()[1]
        output = torch.zeros(batch_size, num_rois, num_channels,
                             self.crop_depth+self.start_pooled_depth+self.end_pooled_depth)

        output = output.cuda()
        _backend.crop_and_resize_gpu_forward(features, rois, self.extrapolation_value,
                                             self.crop_depth, self.temporal_scale, self.start_pooled_depth, 
                                             self.end_pooled_depth, self.bratio, output)
        # self.output = output
        self.rois = rois
        self.feature_size = features.size()

        return output

    def backward(self, grad_output):
        assert(self.feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_depth = self.feature_size

        grad_input = torch.zeros(batch_size, num_channels, data_depth).cuda()
        _backend.crop_and_resize_gpu_backward(
            grad_output, self.rois, self.crop_depth,
            self.crop_height, self.crop_width,
            self.temporal_scale, grad_input
        )
        self.rois = None

        return grad_input, None
