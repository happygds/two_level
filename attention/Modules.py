import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

# from .torchsparseattn.fused import Fusedmax, FusedProxFunction
# from sparsemax import Sparsemax

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
            self.conv_layers = nn.Sequential(
                nn.Conv2d(self.n_head, 8*self.n_head, 3, padding=1),
                nn.BatchNorm2d(
                    8*self.n_head), nn.ReLU(),
                #  nn.Conv2d(8*self.n_head, 8*self.n_head, 3, padding=1),
                #  nn.BatchNorm2d(8*self.n_head), nn.ReLU(),
                nn.Conv2d(8*self.n_head, self.n_head, 3, padding=1),
                nn.BatchNorm2d(self.n_head))
        elif self.kernel_type == 'highorder-causal':
            self.conv_layers = nn.Sequential(
                nn.Conv2d(self.n_head, 8*self.n_head, (3, 1), padding=(1, 0)),
                nn.Conv2d(8*self.n_head, 8*self.n_head, (1, 3), padding=(0, 1)),
                nn.BatchNorm2d(
                    8*self.n_head), nn.ReLU(),
                #  nn.Conv2d(8*self.n_head, 8*self.n_head, 3, padding=1),
                #  nn.BatchNorm2d(8*self.n_head), nn.ReLU(),
                nn.Conv2d(8*self.n_head, 8*self.n_head, (3, 1), padding=(1, 0)),
                nn.Conv2d(8*self.n_head, self.n_head, (1, 3), padding=(0, 1)),
                nn.BatchNorm2d(self.n_head))

    def forward(self, q, k, v, attn_mask=None, attn_pos_emb=None):
        if self.kernel_type == 'self_attn':
            attn = torch.bmm(q, k.transpose(1, 2)) / self.temper
            # out_attn = attn
            if attn_pos_emb is not None:
                k_pos_emb, v_pos_emb = torch.split(attn_pos_emb, q.size(2), dim=3)
                # k_gate = F.sigmoid(torch.mean(k.unsqueeze(1) + k_pos_gate, dim=3))
                # attn = k_gate * attn + (1. - k_gate) * torch.sum(q.unsqueeze(2) * k_pos_emb, dim=3) / self.temper
                attn += torch.sum(q.unsqueeze(2) * k_pos_emb, dim=3) / self.temper
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
        elif self.kernel_type in ['highorder', 'highorder-causal']:
            attn = torch.bmm(q, k.transpose(1, 2)) / self.temper
            if attn_pos_emb is not None:
                k_pos_emb, v_pos_emb = torch.split(attn_pos_emb, q.size(2), dim=3)
                k_gate = F.tanh(torch.mean(k.unsqueeze(1) + k_pos_gate, dim=3))
                # attn = k_gate * attn + (1. - k_gate) * torch.sum(q.unsqueeze(2) * k_pos_emb, dim=3) / self.temper
                attn *= k_gate

            attn.data.masked_fill_(attn_mask, 0)
            attn_reshape = attn.view(
                (self.n_head, -1) + attn.size()[1:]).transpose(0, 1).contiguous()
            # conv_attn_mask = attn_mask.view((self.n_head, -1) + attn.size()[1:]).transpose(0, 1).contiguous()
            # attn_reshape.data.masked_fill_(conv_attn_mask, 0)
            conv_attn = self.conv_layers(attn_reshape)
            attn = conv_attn.transpose(
                0, 1).contiguous().view(attn.size()) + attn
        elif self.kernel_type in ['roi_remov']:
            attn = torch.bmm(q, k.transpose(1, 2)) / self.temper
            assert attn_pos_emb is not None
            attn.data.masked_fill_(attn_mask, -1e+32)
            attn_max = torch.max(attn, 2, keepdim=True)[0]
            attn = torch.exp(attn - attn_max)
            attn.data.masked_fill_(attn_mask, 0)
            attn = attn_pos_emb * attn / torch.sum(attn_pos_emb * attn, 2, keepdim=True).clamp(1e-14)
        else:
            raise NotImplementedError()

        # attn /= 0.1
        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), \
                'Attention mask shape {} mismatch ' \
                'with Attention logit tensor shape ' \
                '{}.'.format(attn_mask.size(), attn.size())
            if self.kernel_type in ['self_attn', 'addition', 'inner_prod', 'highorder', 'highorder-causal']:
                attn.data.masked_fill_(attn_mask, -float('inf'))
                # attn.data.masked_fill_(attn_mask, -1e+32)
            else:
                attn.data.masked_fill_(attn_mask, 0)

        if self.kernel_type in ['self_attn', 'addition', 'inner_prod', 'highorder', 'highorder-causal']:
            attn = self.softmax(attn)
            attn.data.masked_fill_(torch.isnan(attn), 0)
            # shp = attn.size()
            # lengths = (1. - attn_mask)[:, 0].sum(-1).long().cuda()
            # attn = self.softmax(attn.data.cpu(), lengths.data.cpu()).view(shp).cuda()
        else:
            attn = attn / attn.sum(dim=2, keepdim=True).clamp(1e-14)
        out_attn = attn
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        if attn_pos_emb is not None and self.kernel_type not in ['roi_remov']:
            # v_gate = F.sigmoid(torch.mean(v_pos_emb + v.unsqueeze(1), dim=2))
            # output = v_gate * output + (1. - v_gate) * torch.sum(attn.unsqueeze(3) * v_pos_emb, dim=2)
            output += torch.sum(attn.unsqueeze(3) * v_pos_emb, dim=2)

        return output, out_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, d_out=None, 
                 dropout=0.1, kernel_type='self_attn', groupwise_heads=0):
        super(MultiHeadAttention, self).__init__()
        self.groupwise_heads = groupwise_heads
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
        if kernel_type == 'roi_remov':
            self.w_rs = nn.Parameter(torch.FloatTensor(n_head, d_model, 1))
            init.xavier_normal_(self.w_rs)
            self.b_rs = nn.Parameter(torch.FloatTensor(n_head, 1, 1))
            init.xavier_normal_(self.b_rs)
        self.kernel_type = kernel_type

    def forward(self, q, k, v, attn_mask=None, attn_pos_emb=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        residual = q

        mb_size, len_q, d_model = q.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()

        if attn_pos_emb is not None:
            attn_pos_emb = attn_pos_emb.repeat(n_head, 1, 1, 1)
        if self.kernel_type == 'roi_remov':
            attn_pos_emb = attn_pos_emb.view(n_head, -1, d_model)
            attn_pos_emb = F.relu((torch.bmm(attn_pos_emb, self.w_rs) + self.b_rs).view(-1, len_q, len_k))

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
        if self.groupwise_heads > 0:
            trn_kernel = self.groupwise_heads
            assert self.n_head % 4 == 0
            k_head = self.n_head // 4 * mb_size
            q_s1, q_s2, q_s3, q_s4 = torch.split(q_s, k_head, dim=0)
            q_s2 = F.avg_pool1d(F.pad(q_s2.transpose(1, 2), (trn_kernel-1, 0)), trn_kernel, stride=1).transpose(1, 2)
            q_s3 = F.avg_pool1d(F.pad(q_s3.transpose(1, 2), (0, trn_kernel-1)), trn_kernel, stride=1).transpose(1, 2)
            q_s4 = F.avg_pool1d(F.pad(q_s4.transpose(1, 2), ((trn_kernel-1)//2, (trn_kernel-1)//2)), trn_kernel, stride=1).transpose(1, 2)
            q_s = torch.cat([q_s1, q_s2, q_s3, q_s4], dim=0)

        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(n_head, 1, 1)
        outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=attn_mask, attn_pos_emb=attn_pos_emb)

        # back to original mb_size batch, result size = mb_size x len_v x (n_head*d_v)
        # outputs = outputs - v_s
        outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1)
        # (n_head*mb_size) x len_q x len_k -> mb_size x n_head x len_q x len_k
        attns = [x.unsqueeze(1) for x in torch.split(attns, mb_size, dim=0)]
        attns = torch.cat(attns, dim=1)

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
