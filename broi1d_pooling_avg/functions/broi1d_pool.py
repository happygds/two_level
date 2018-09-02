import torch
import numpy as np
from torch.autograd import Function
from .._ext import broi1d_pooling


class BRoI1DPoolFunction(Function):
    def __init__(self, pooled_depth, temporal_scale, start_pooled_depth, end_pooled_depth, bratio):
        self.pooled_depth = int(pooled_depth)
        self.temporal_scale = float(temporal_scale)
        self.start_pooled_depth = int(start_pooled_depth)
        self.end_pooled_depth = int(end_pooled_depth)
        self.bratio = float(bratio)
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        batch_size, num_channels, data_depth = features.size()
        num_rois = rois.size()[1]
        output = torch.zeros(batch_size, num_rois, num_channels,
                             self.pooled_depth+self.start_pooled_depth+self.end_pooled_depth)

        output = output.cuda()
        broi1d_pooling.broi1d_pooling_forward_cuda(self.pooled_depth, self.temporal_scale, self.start_pooled_depth, 
                                                   self.end_pooled_depth, self.bratio, features, rois, output)
        import pdb; pdb.set_trace()
        # output.data.masked_fill_(torch.isnan(output), 0)
        self.rois = rois
        self.feature_size = features.size()

        return output

    def backward(self, grad_output):
        assert(self.feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_depth = self.feature_size

        grad_input = torch.zeros(batch_size, num_channels, data_depth).cuda()
        broi1d_pooling.broi1d_pooling_backward_cuda(self.pooled_depth, self.temporal_scale, self.start_pooled_depth,
                                                    self.end_pooled_depth, self.bratio, grad_output, self.rois, grad_input)
        if np.isnan(grad_input.data.cpu().numpy()).any():
            import pdb; pdb.set_trace()
        self.rois = None

        return grad_input, None
