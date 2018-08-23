from torch.nn.modules.module import Module
from ..functions.roi1d_pool import RoI1DPoolFunction


class RoI1DPool(Module):
    def __init__(self, pooled_depth, temporal_scale):
        super(RoI1DPool, self).__init__()

        self.pooled_depth = int(pooled_depth)
        self.temporal_scale = float(temporal_scale)

    def forward(self, features, rois):
        return RoI1DPoolFunction(self.pooled_depth, self.temporal_scale)(features, rois)
