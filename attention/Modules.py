import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

# from .torchsparseattn.fused import Fusedmax, FusedProxFunction
from sparsemax import Sparsemax


class CE_Criterion(nn.Module):
    def __init__(self, use_weight=True, gamma=0.1):
        super(CE_Criterion, self).__init__()
        # self.gamma = gamma
        self.use_weight = use_weight

    def forward(self, x, target, weight=None, mask=None):
        # for i, x in enumerate(inputs):
        output = - target * torch.log(x)
        # output = output * (1. - x) ** self.gamma
        if self.use_weight:
            output *= weight.unsqueeze(1)
            output = torch.sum(output.mean(2) * mask, dim=1) / torch.sum(mask, dim=1)
            # output = torch.sum(output.mean(2) * mask) / torch.sum(mask)
        return torch.mean(output)


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
        #     self.split = nn.Conv2d(self.n_head, 3*self.n_head, 1)

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
            attn.data.masked_fill_(attn_mask, -1e+13)
            qsize = q.size()
            topk_inds = torch.topk(attn, num_local, dim=2)[1].unsqueeze(
                3).expand(-1, -1, -1, qsize[2]).requires_grad_(False)
            import pdb
            pdb.set_trace()
            q_topk, k_topk, v_topk = q.unsqueeze(2).expand(-1, -1, qsize[1], -1), \
                k.unsqueeze(2).expand(-1, -1, qsize[1], -1), \
                v.unsqueeze(2).expand(-1, -1, qsize[1], -1)
            q_topk, k_topk, v_topk = torch.gather(q_topk, 2, topk_inds).view(qsize[0], qsize[1]*num_local, qsize[2]), \
                torch.gather(k_topk, 2, topk_inds).view(qsize[0], qsize[1]*num_local, qsize[2]), \
                torch.gather(v_topk, 2, topk_inds).view(qsize[0], qsize[1]*num_local, qsize[2])
            attn_topk = torch.bmm(q_topk, k_topk.transpose(1, 2)) / self.temper
            attn_topk_mask = attn_mask[:, 0].unsqueeze(2).expand(-1, -1, num_local).contiguous(
            ).view(qsize[0], qsize[1]*num_local).unsqueeze(1).expand(attn_topk.size())
            attn_topk_mask = torch.gt(
                attn_topk_mask + attn_topk_mask.transpose(1, 2), 0)
            attn_topk.data.masked_fill_(attn_topk_mask, -float('inf'))
            attn_topk = self.dropout(F.softmax(attn_topk, dim=2))
            attn_topk.data.masked_fill_(torch.isnan(attn_topk), 0)
            attn_topk = torch.bmm(attn_topk, v_topk).view(
                qsize[0], qsize[1], num_local, qsize[2]).mean(2)
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
        if self.kernel_type in ['highorder-nonlocal']:
            output += attn_topk

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
