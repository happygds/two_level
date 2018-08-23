import torch
import torch.nn as nn
from torch.autograd import Variable
from modules.roi3d_pool import RoI3DPool

spatial_scale = 1. / 16
temporal_scale = spatial_scale
group_size = 7
output_dim = 392


indata = torch.randn(2, 3, 128, 224, 224)
indata = indata.cuda()
indata = Variable(indata)


rois = torch.FloatTensor([[[0, 0, 32], [0, 25, 64]], [[1, 0, 32], [1, 25, 64]]])
rois = rois.cuda()
inrois = Variable(rois)

roi3d_pooling = RoI3DPool(2, 1, 1, spatial_scale, temporal_scale)
print(roi3d_pooling)
# raw_input()
output = roi3d_pooling.forward(indata, inrois)
indata = indata.data.cpu().numpy()
print(output.squeeze().data.cpu().numpy(), indata[1, :, 2:5, 0:7, 0:7].max(axis=(2, 3)))
