import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

from .torchsparseattn.fused import Fusedmax

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class CE_Criterion(nn.Module):
    def __init__(self, use_weight=False):
        super(CE_Criterion, self).__init__()
        self.use_weight = use_weight

    def forward(self, x, target, weight=None, mask=None, lambda_l=1.):
        output = - target * torch.log(x)
        if self.use_weight:
            output *= weight.unsqueeze(0).unsqueeze(0)
            # output = torch.sum(output.mean(2) * mask, dim=1) / torch.sum(mask, dim=1)
            output = torch.sum(output.mean(2) * mask) / torch.sum(mask)
        return torch.mean(output)

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_model, d_k, attn_dropout=0.1, kernel_type='self_attn'):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        # self.softmax = nn.Softmax(dim=2)
        self.softmax = Fusedmax()
        self.kernel_type = kernel_type
        if self.kernel_type == 'concat':
            self.fc1 = nn.Linear(d_k, 1)
            self.fc2 = nn.Linear(d_k, 1)
            self.relu = nn.ReLU()
        elif self.kernel_type == 'addition':
            self.fc = nn.Sequential(nn.Tanh(), nn.Linear(d_k, 1))

    def forward(self, q, k, v, attn_mask=None):
        if self.kernel_type == 'self_attn':
            attn = torch.bmm(q, k.transpose(1, 2)) / self.temper
        elif self.kernel_type == 'dot':
            attn = torch.bmm(q, k.transpose(1, 2))
        elif self.kernel_type == 'concat':
            attn = self.fc1(q) + self.fc2(k).transpose(1, 2)
            attn = self.relu(attn)
        elif self.kernel_type == 'addition':
            len_q, len_k = q.size(1), k.size(1)
            q = q.unsqueeze(2)
            k = k.unsqueeze(1)
            attn = self.fc(q + k).squeeze(3)
        else:
            raise NotImplementedError()

        if attn_mask is not None:

            assert attn_mask.size() == attn.size(), \
                    'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape ' \
                    '{}.'.format(attn_mask.size(), attn.size())
            if self.kernel_type in ['self_attn', 'addition']:
                attn.data.masked_fill_(attn_mask, -float('inf'))
                # attn.data.masked_fill_(attn_mask, -1e+32)
            else:
                attn.data.masked_fill_(attn_mask, 0)

        if self.kernel_type in ['self_attn', 'addition']:
            # attn = self.softmax(attn)
            shp = attn.size()
            lengths = (1. - attn_mask).sum(-1).long().cuda().view(-1)
            attn = self.softmax(attn.view(-1, shp[2]).data.cpu(), lengths.data.cpu()).view(shp).cuda()
            import pdb
            pdb.set_trace()
        else:
            attn = attn / attn.sum(dim=2, keepdim=True).clamp(1e-14)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn
