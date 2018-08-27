import torch
from torch.autograd import Function
from .._ext import roi1d_pooling


class RoI1DPoolFunction(Function):
    def __init__(self, pooled_depth, temporal_scale=1.):
        self.pooled_depth = int(pooled_depth)
        self.temporal_scale = float(temporal_scale)
        self.output = None
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        batch_size, num_channels, data_depth = features.size()
        num_rois = rois.size()[1]
        output = torch.zeros(batch_size, num_rois, num_channels,
                             self.pooled_depth)

        # if not features.is_cuda:
        #     _features = features.permute(0, 2, 3, 1)
        #     roi3d_pooling.roi3d_pooling_forward(self.pooled_height, self.pooled_width, self.spatial_scale,
        #                                     _features, rois, output)
        #     # output = output.cuda()
        # else:
        output = output.cuda()
        roi1d_pooling.roi1d_pooling_forward_cuda(self.pooled_depth, self.temporal_scale, features, rois, output)
        self.output = output
        self.rois = rois
        self.feature_size = features.size()
        del output, rois

        return output

    def backward(self, grad_output):
        assert(self.feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_depth = self.feature_size

        grad_input = torch.zeros(batch_size, num_channels, data_depth).cuda()
        roi1d_pooling.roi1d_pooling_backward_cuda(self.pooled_depth, self.temporal_scale, grad_output,
                                                  self.rois, grad_input)

        # print grad_input

        return grad_input, None
