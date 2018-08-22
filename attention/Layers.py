''' Define the Layers '''
import torch
import torch.nn as nn
import torch.nn.init as init
# from .sparsemax import Sparsemax
from .Modules import ScaledDotProductAttention, MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, 
                 dropout=0.1, kernel_type='self_attn', groupwise_heads=0):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, 
            kernel_type=kernel_type, groupwise_heads=groupwise_heads)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner_hid, dropout=dropout)

    def forward(self, enc_input, local_attn_mask=None, 
                slf_attn_mask=None, attn_pos_emb=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input,
            attn_mask=slf_attn_mask, attn_pos_emb=attn_pos_emb)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class Local_EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, 
                 dropout=0.1, kernel_type='self_attn', 
                 local_type=None):
        super(Local_EncoderLayer, self).__init__()
        self.local_type = local_type
        self.local_attn = MultiHeadAttention(n_head//4, d_model, d_k*4, d_v*4, 
                                             dropout=dropout, kernel_type='self_attn')
        if self.local_type == 'diff':
            self.layernorm = nn.LayerNorm(d_model)

        # for non-local operation
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, 
                                           dropout=dropout, kernel_type=kernel_type)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner_hid, dropout=dropout)

    def forward(self, enc_input, local_attn_mask=None, 
                slf_attn_mask=None, attn_pos_emb=None):
        local_output, local_attn = self.local_attn(
            enc_input, enc_input, enc_input,
            attn_mask=local_attn_mask, attn_pos_emb=attn_pos_emb)

        if self.local_type == 'qkv':
            enc_output, enc_slf_attn = self.slf_attn(
                local_output, local_output, local_output, 
                attn_mask=slf_attn_mask, attn_pos_emb=attn_pos_emb)
        elif self.local_type == 'qv':
            enc_output, enc_slf_attn = self.slf_attn(
                local_output, enc_input, local_output, 
                attn_mask=slf_attn_mask, attn_pos_emb=attn_pos_emb)
        elif self.local_type == 'kv':
            enc_output, enc_slf_attn = self.slf_attn(
                enc_input, local_output, local_output, 
                attn_mask=slf_attn_mask, attn_pos_emb=attn_pos_emb)
        elif self.local_type == 'diff':
            enc_input = self.layernorm(enc_input)
            enc_output, enc_slf_attn = self.slf_attn(
                local_output - enc_input, local_output - enc_input, 
                local_output - enc_input, attn_mask=slf_attn_mask)
            enc_output = enc_output + enc_input
        else:
            raise NotImplementedError()
        # enc_output = self.pos_ffn_slf(enc_output)

        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

