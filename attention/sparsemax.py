from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Sparsemax(nn.Module):
    def __init__(self, mask_value=None):
        super(Sparsemax, self).__init__()
        self.mask_value = mask_value
        # self.num_clusters = num_clusters
        # self.num_neurons_per_cluster = num_neurons_per_cluster
        
    def forward(self, input):

        input_reshape = torch.zeros(input.size())
        self.num_clusters, self.num_neurons_per_cluster = input.size()[1:]
        input_reshape = input.view(-1, self.num_clusters, self.num_neurons_per_cluster)
        dim = 2
        #translate for numerical stability
        input_shift = input_reshape # - torch.max(input_reshape, dim)[0].expand_as(input_reshape)

        #sorting input in descending order
        z_sorted = torch.sort(input_shift, dim=dim, descending=True)[0]
        input_size = input_shift.size()[dim]	
        range_values = torch.arange(1, input_size+1).cuda()
        range_values = range_values.expand_as(z_sorted)
        if self.mask_value is not None:
            z_mask = torch.ne(z_sorted, self.mask_value).float()

        #Determine sparsity of projection
        bound = torch.zeros(z_sorted.size()).cuda()
        bound = 1 + torch.addcmul(bound, range_values, z_sorted)
        cumsum_zs = torch.cumsum(z_sorted, dim)
        is_gt = torch.gt(bound, cumsum_zs).type(torch.FloatTensor).cuda()
        if self.mask_value is not None:
            is_gt = is_gt * z_mask
        valid = torch.zeros(range_values.size()).cuda()
        valid = torch.addcmul(valid, range_values, is_gt)
        k_max = torch.max(valid, dim)[0]
        zs_sparse = torch.zeros(z_sorted.size()).cuda()
        zs_sparse = torch.addcmul(zs_sparse, is_gt, z_sorted)
        sum_zs = (torch.sum(zs_sparse, dim) - 1)
        taus = torch.zeros(k_max.size()).cuda()
        taus = torch.addcdiv(taus, (torch.sum(zs_sparse, dim) - 1), k_max)
        # print(taus.size(), input_reshape.size())
        taus_expanded = taus.unsqueeze(2).expand_as(input_reshape)
        output = Variable(torch.zeros(input_reshape.size())).cuda()
        output = torch.max(output, input_shift - taus_expanded)
        # return output.view(-1, self.num_clusters*self.num_neurons_per_cluster), zs_sparse,taus, is_gt
        self.output = output
        return output.view(-1, self.num_clusters, self.num_neurons_per_cluster)
		 

    def backward(self, grad_output):
        self.num_clusters, self.num_neurons_per_cluster = grad_output.size()[1:]
        self.output = self.output.view(-1,self.num_clusters, self.num_neurons_per_cluster)
        grad_output = grad_output.view(-1,self.num_clusters, self.num_neurons_per_cluster)
        dim = 2
        non_zeros = torch.ne(self.output, 0).type(torch.FloatTensor).cuda()
        mask_grad = torch.zeros(self.output.size()).cuda()
        mask_grad = torch.addcmul(mask_grad, non_zeros, grad_output)
        sum_mask_grad = torch.sum(mask_grad, dim)
        l1_norm_non_zeros = torch.sum(non_zeros, dim)
        sum_v = torch.zeros(sum_mask_grad.size()).cuda()
        sum_v = torch.addcdiv(sum_v, sum_mask_grad, l1_norm_non_zeros)
        self.gradInput = Variable(torch.zeros(grad_output.size()))
        self.gradInput = torch.addcmul(self.gradInput, non_zeros, grad_output - sum_v.expand_as(grad_output))
        self.gradInput = self.gradInput.view(-1, self.num_clusters, self.num_neurons_per_cluster)
        return self.gradInput