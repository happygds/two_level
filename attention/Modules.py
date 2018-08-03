import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

# from .torchsparseattn.fused import Fusedmax, FusedProxFunction
from sparsemax import Sparsemax

def convert_categorical(x_in, n_classes=3):
    shp = x_in.shape
    x = (x_in.ravel().astype('int'))
    x_mask = (x >= 0).reshape(-1, 1)
    x = x.clip(0)
    y = np.diag(np.ones((n_classes,)))
    y = y[x] * x_mask
    y = y.reshape(shp + (n_classes,)).astype('float32')
    return y

class CE_Criterion(nn.Module):
    def __init__(self, use_weight=True, l_step=1.1):
        super(CE_Criterion, self).__init__()
        self.l_step = l_step
        self.use_weight = use_weight

    def forward(self, inputs, target, weight=None, mask=None, diff_score=None):
        if isinstance(inputs, list):
            for i, x in enumerate(inputs):
                if i == 0:
                    output = - target * torch.log(x)
                else:
                    output += - target * torch.log(x) * self.l_step ** i
        else:
            output = - target * torch.log(inputs)
     
        if self.use_weight:
            output *= weight.unsqueeze(1)
            output = torch.sum(output.mean(2) * mask, dim=1) / \
                torch.sum(mask, dim=1)
            output = torch.mean(output)
        
        if diff_score is not None:
            target_ind = target.max(2)[1].float()
            diff_mask = mask[:, 1:]
            target_diff = convert_categorical((target_ind[:, 1:] - target_ind[:, :-1] + 1).cpu().numpy(), n_classes=3)
            target_diff = torch.from_numpy(target_diff).cuda().requires_grad_(False)
            target_diff *= diff_mask.unsqueeze(2)
            diff_weight = target_diff.sum(1) / diff_mask.sum(1).unsqueeze(1)
            diff_weight = 1./ 3 / diff_weight.clamp(0.001)
            diff_output = - target_diff * torch.log(diff_score) * diff_weight.unsqueeze(1)
            diff_output = torch.sum(diff_output.mean(2) * diff_mask, dim=1) / \
                torch.sum(diff_mask, dim=1).clamp(0.001)
            diff_output = torch.mean(diff_output)
        else:
            target_diff = 1. - (target[:, 1:, :] - target[:, :-1, :]).abs().max(2)[0]
            assert not isinstance(inputs, list)
            mask_diff = mask[:, 1:]
            inputs_diff = (inputs[:, 1:, :] - inputs[:, :-1, :]).abs().mean(2)
            diff_output = torch.sum(inputs_diff * mask_diff * target_diff, dim=1) / torch.sum(mask_diff * target_diff, dim=1).clamp(0.001)
            diff_output += 1. - (torch.sum(inputs_diff * mask_diff * (1. - target_diff), dim=1) / torch.sum(mask_diff * (1. - target_diff), dim=1).clamp(0.001)).clamp(0.001)
            diff_output = torch.mean(diff_output)

        return output, diff_output


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_model, d_k, n_head, attn_dropout=0.1, kernel_type='self_attn'):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.n_head = n_head
        self.softmax = nn.Softmax(dim=2)
        # self.softmax = Sparsemax(mask_value=-1e+32)
        self.kernel_type = kernel_type
        if self.kernel_type == 'concat':
            self.fc1 = nn.Linear(d_k, 1)
            self.fc2 = nn.Linear(d_k, 1)
            self.relu = nn.ReLU()
        elif self.kernel_type == 'addition':
            self.fc = nn.Sequential(nn.Tanh(), nn.Linear(d_k, 1))
        elif self.kernel_type == 'highorder':
            self.conv_layers = nn.Sequential(nn.Conv2d(self.n_head, 8*self.n_head, 3, padding=1),
                                             nn.BatchNorm2d(
                                                 8*self.n_head), nn.ReLU(),
                                             #  nn.Conv2d(8*self.n_head, 8*self.n_head, 3, padding=1),
                                             #  nn.BatchNorm2d(8*self.n_head), nn.ReLU(),
                                             nn.Conv2d(
                                                 8*self.n_head, self.n_head, 3, padding=1),
                                             nn.BatchNorm2d(self.n_head))
        # elif self.kernel_type == 'highorder-nonlocal':
        #     self.reduce = nn.Linear(4*d_k, d_k)

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
        elif self.kernel_type == 'inner_prod':
            attn = torch.bmm(q, k.transpose(1, 2)) / \
                (q * q).sum(2).unsqueeze(1)
        elif self.kernel_type == 'highorder':
            attn = torch.bmm(q, k.transpose(1, 2)) / self.temper
            # print(attn.mean(), attn.std())
            attn.data.masked_fill_(attn_mask, 0)
            attn_reshape = attn.view(
                (self.n_head, -1) + attn.size()[1:]).transpose(0, 1).contiguous()
            # conv_attn_mask = attn_mask.view((self.n_head, -1) + attn.size()[1:]).transpose(0, 1).contiguous()
            # attn_reshape.data.masked_fill_(conv_attn_mask, 0)
            conv_attn = self.conv_layers(attn_reshape)
            attn = conv_attn.transpose(
                0, 1).contiguous().view(attn.size()) + attn
        elif self.kernel_type == 'highorder-nonlocal':
            num_local = 3
            attn = torch.bmm(q, k.transpose(1, 2)) / self.temper
            q_corr, k_corr = torch.bmm(q, q.transpose(1, 2)) / self.temper, torch.bmm(k, k.transpose(1, 2)) / self.temper
            qsize = attn.size()
            high_attn = torch.bmm(q_corr, k_corr.transpose(1, 2)) / np.power(qsize[1], 0.5)
            attn += high_attn
        else:
            raise NotImplementedError()

        # attn /= 0.1
        if attn_mask is not None:

            assert attn_mask.size() == attn.size(), \
                'Attention mask shape {} mismatch ' \
                'with Attention logit tensor shape ' \
                '{}.'.format(attn_mask.size(), attn.size())
            if self.kernel_type in ['self_attn', 'addition', 'inner_prod', 'highorder', 'highorder-nonlocal']:
                attn.data.masked_fill_(attn_mask, -float('inf'))
                # attn.data.masked_fill_(attn_mask, -1e+32)
            else:
                attn.data.masked_fill_(attn_mask, 0)

        if self.kernel_type in ['self_attn', 'addition', 'inner_prod', 'highorder', 'highorder-nonlocal']:
            attn = self.softmax(attn)
            attn.data.masked_fill_(torch.isnan(attn), 0)
            # shp = attn.size()
            # lengths = (1. - attn_mask)[:, 0].sum(-1).long().cuda()
            # attn = self.softmax(attn.data.cpu(), lengths.data.cpu()).view(shp).cuda()
        else:
            attn = attn / attn.sum(dim=2, keepdim=True).clamp(1e-14)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, d_out=None, dropout=0.1, kernel_type='self_attn'):
        super(MultiHeadAttention, self).__init__()
        self.d_out = d_out
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

        self.attention = ScaledDotProductAttention(
            d_model, d_k, n_head, attn_dropout=dropout, kernel_type=kernel_type)
        self.layer_norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(n_head*d_v, d_model)
        if self.d_out is not None:
            self.proj_cluster = nn.Linear(n_head*d_v, d_out)

        self.dropout = nn.Dropout(dropout)

        init.xavier_normal_(self.w_qs)
        init.xavier_normal_(self.w_ks)
        init.xavier_normal_(self.w_vs)

    def forward(self, q, k, v, attn_mask=None):

        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        residual = q

        mb_size, len_q, d_model = q.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()

        # treat as a (n_head) size batch
        # n_head x (mb_size*len_q) x d_model
        q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model)
        # n_head x (mb_size*len_k) x d_model
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model)
        # n_head x (mb_size*len_v) x d_model
        v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model)

        # treat the result as a (n_head * mb_size) size batch
        # (n_head*mb_size) x len_q x d_k
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)
        # (n_head*mb_size) x len_k x d_k
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)
        # (n_head*mb_size) x len_v x d_v
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)

        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(n_head, 1, 1)
        outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=attn_mask)

        # back to original mb_size batch, result size = mb_size x len_v x (n_head*d_v)
        # outputs = outputs - v_s
        outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1)
        # # (n_head*mb_size) x len_q x len_k -> (n_head*mb_size) x len_k -> mb_size x len_k x n_head
        # attns = torch.cat(torch.split(attns.mean(1), mb_size, dim=0), dim=-1).view(-1, len_k, n_head)

        # project back to residual size
        if self.d_out is not None:
            cluster_outputs = self.dropout(self.proj_cluster(outputs))
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)

        if self.d_out is None:
            return self.layer_norm(outputs + residual), attns
        else:
            return self.layer_norm(outputs), cluster_outputs


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_hid, d_inner_hid, d_in=None, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.d_in = d_in
        if d_in is None:
            d_in = d_hid
        self.w_1 = nn.Linear(d_in, d_inner_hid)  # position-wise
        self.w_2 = nn.Linear(d_inner_hid, d_hid)  # position-wise
        self.layer_norm = nn.LayerNorm(d_hid)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.w_1(x))
        output = self.w_2(output)
        output = self.dropout(output)
        if self.d_in is None:
            return self.layer_norm(output + residual)
        else:
            return self.layer_norm(output)
