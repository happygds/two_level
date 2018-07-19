''' Define the Layers '''
import torch
import torch.nn as nn
import torch.nn.init as init
from .Modules import ScaledDotProductAttention, MultiHeadAttention, PositionwiseFeedForward


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

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1, kernel_type='self_attn', local_type=None):
        super(Local_EncoderLayer, self).__init__()
        self.local_type = local_type
        self.local_attn = MultiHeadAttention(
            n_head//4, d_model, d_k*4, d_v*4, dropout=dropout, kernel_type=kernel_type)

        # for non-local operation
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, kernel_type=kernel_type)
        self.concat = nn.Linear(2 * d_model, d_model)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner_hid, dropout=dropout)

    def forward(self, enc_input, local_attn_mask=None, slf_attn_mask=None):
        local_output, local_attn = self.local_attn(
            enc_input, enc_input, enc_input, attn_mask=local_attn_mask)

        if self.local_type == 'qkv':
            enc_output, enc_slf_attn = self.slf_attn(
            local_output, local_output, local_output, attn_mask=slf_attn_mask)
        elif self.local_type == 'kv':
            enc_output, enc_slf_attn = self.slf_attn(
                enc_input, local_output, local_output, attn_mask=slf_attn_mask)
        elif self.local_type == 'v':
            enc_output, enc_slf_attn = self.slf_attn(
                enc_input, enc_input, local_output, attn_mask=slf_attn_mask)
        elif self.local_type == 'qv':
            enc_output, enc_slf_attn = self.slf_attn(
                local_output, enc_input, local_output, attn_mask=slf_attn_mask)
        elif self.local_type == 'qk':
            enc_output, enc_slf_attn = self.slf_attn(
                local_output, local_output, enc_input, attn_mask=slf_attn_mask)
        elif self.local_type == 'q':
            enc_output, enc_slf_attn = self.slf_attn(
                local_output, enc_input, enc_input, attn_mask=slf_attn_mask)
        else:
            raise NotImplementedError()

        enc_output = self.concat(torch.cat((local_output, enc_output), dim=2))            
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class Cluster_EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1, kernel_type='self_attn', n_cluster=64, local_type=None):
        super(Cluster_EncoderLayer, self).__init__()
        # self.local_type = local_type
        # self.local_attn = MultiHeadAttention(
        #     n_head//4, d_model, d_k*4, d_v*4, dropout=dropout, kernel_type=kernel_type)

        self.assign_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, d_out=n_cluster, dropout=dropout, kernel_type=kernel_type)
        self.assign_softmax = nn.Softmax(dim=1)

        # for non-local operation
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, kernel_type=kernel_type)
        self.pos_ffn_slf = PositionwiseFeedForward(
            d_model, d_inner_hid, dropout=dropout)
        # for non-local operation
        self.slf_attn_cluster = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, kernel_type=kernel_type)
        self.pos_ffn_cluster = PositionwiseFeedForward(
            d_model, d_inner_hid, dropout=dropout)
        
        self.concat = nn.Linear(2 * d_model, d_model)

    def forward(self, enc_input, local_attn_mask=None, slf_attn_mask=None):
        assign_mat, _ = self.assign_attn(
            enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)
        assign_mask = (1. - slf_attn_mask[:, 0].float()).unsqueeze(2).expand(assign_mat.size()).byte()
        assign_mat.data.masked_fill_(assign_mask, -float('inf'))
        assign_mat = self.assign_softmax(assign_mat)   # mb_size * len_q * n_cluster
        import pdb
        pdb.set_trace()

        enc_slf_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)
        enc_slf_output = self.pos_ffn_slf(enc_slf_output)
        cluster_input = torch.bmm(assign_mat.transpose(1, 2), enc_slf_output)

        cluster_output, _ = self.slf_attn(
            cluster_input, cluster_input, cluster_input)
        cluster_output = self.pos_ffn_cluster(cluster_output)
        cluster_output = torch.bmm(assign_mat, cluster_output)
        

        enc_output = self.concat(torch.cat((cluster_output, enc_slf_output), dim=2))            
        # enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

