from torch.nn.modules.module import Module
from ..functions.broi1d_pool import BRoI1DPoolFunction


class BRoI1DPool(Module):
    def __init__(self, pooled_depth, temporal_scale, start_pooled_depth, end_pooled_depth, bratio=1./5):
        super(BRoI1DPool, self).__init__()
        self.pooled_depth = int(pooled_depth)
        self.temporal_scale = float(temporal_scale)
        self.start_pooled_depth = int(start_pooled_depth)
        self.end_pooled_depth = int(end_pooled_depth)
        self.bratio = float(bratio)

    def forward(self, features, rois):
        return BRoI1DPoolFunction(self.pooled_depth, self.temporal_scale, self.start_pooled_depth, 
                                  self.end_pooled_depth, self.bratio)(features, rois)
