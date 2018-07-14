''' Define the Layers '''
import torch
import torch.nn as nn
import torch.nn.init as init
from .Modules import ScaledDotProductAttention



class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, kernel_type='self_attn'):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

        self.attention = ScaledDotProductAttention(
            d_model, d_k, attn_dropout=dropout, kernel_type=kernel_type)
        self.layer_norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(n_head*d_v, d_model)

        self.dropout = nn.Dropout(dropout)

        init.xavier_normal_(self.w_qs)
        init.xavier_normal_(self.w_ks)
        init.xavier_normal_(self.w_vs)

    def forward(self, q, k, v, attn_mask=None):

        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        residual = v

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
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)

        return self.layer_norm(outputs + residual), attns


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_hid, d_inner_hid)  # position-wise
        self.w_2 = nn.Linear(d_inner_hid, d_hid)  # position-wise
        self.layer_norm = nn.LayerNorm(d_hid)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.w_1(x))
        output = self.w_2(output)
        output = self.dropout(output)
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1, kernel_type='self_attn'):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, kernel_type=kernel_type)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner_hid, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class Local_EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1, kernel_type='self_attn'):
        super(Local_EncoderLayer, self).__init__()
        self.local_attn = MultiHeadAttention(
            n_head//4, d_model, d_k*4, d_v*4, dropout=dropout, kernel_type=kernel_type)

        # for non-local operation
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, kernel_type=kernel_type)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner_hid, dropout=dropout)

    def forward(self, enc_input, local_attn_mask=None, slf_attn_mask=None):
        local_output, local_attn = self.local_attn(
            enc_input, enc_input, enc_input, attn_mask=local_attn_mask)
        enc_output, enc_slf_attn = self.slf_attn(
            local_output, enc_input, enc_input, attn_mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn
