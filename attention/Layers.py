''' Define the Layers '''
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
# from .sparsemax import Sparsemax
from .Modules import ScaledDotProductAttention, MultiHeadAttention, PositionwiseFeedForward
from .utils import rank_embedding
from roi1d_pooling_avg.modules.roi1d_pool import RoI1DPool


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


class ROI_Relation(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, roipool_size, d_inner_hid, n_head, 
                 d_k, d_v, dropout=0.1, kernel_type='roi_remov'):
        super(ROI_Relation, self).__init__()
        self.roi_pool = RoI1DPool(roipool_size, 1.)
        start_pool_size = 1
        self.start_pool, self.end_pool = RoI1DPool(start_pool_size, 1.), RoI1DPool(start_pool_size, 1.)
        self.instance_norm = nn.InstanceNorm1d(2*start_pool_size+roipool_size)
        self.roi_fc = nn.Sequential(nn.Linear(d_model*(2*start_pool_size+roipool_size), d_model), nn.SELU())

        # self.rank_fc = nn.Linear(d_model, d_model)
        # for non-local operation
        # self.slf_attn = MultiHeadAttention(
        #     n_head, d_model, d_k, d_v, dropout=dropout, kernel_type=kernel_type)
        # self.pos_ffn = PositionwiseFeedForward(
        #     d_model, d_inner_hid, dropout=dropout)

    def forward(self, features, start_rois, end_rois, rois, rois_mask, rois_pos_emb):
        inner_feats = self.roi_pool(features.transpose(1, 2), rois)
        start_feats = self.start_pool(features.transpose(1, 2), start_rois)
        end_feats = self.end_pool(features.transpose(1, 2), end_rois)
        roi_feats = torch.cat([start_feats, inner_feats, end_feats], dim=3).transpose(2, 3)
        roi_feat_size = roi_feats.size()
        roi_feats = self.instance_norm(roi_feats.view((-1,)+roi_feat_size[2:])).view(roi_feat_size[:2]+(-1,))

        roi_feats = self.roi_fc(roi_feats)
        if np.isnan(roi_feats.data.cpu().numpy()).any():
            import pdb; pdb.set_trace()

        # compute mask
        mb_size, len_k = roi_feats.size()[:2]
        rois_attn_mask = (1. - rois_mask).unsqueeze(1).expand(mb_size, len_k, len_k).byte()
        rois_attn_mask = torch.gt(rois_attn_mask + rois_attn_mask.transpose(1, 2), 0)
        # use rank embedding
        rank_emb = torch.arange(roi_feat_size[1]).view((1, -1)).float().cuda().requires_grad_(False).expand(roi_feat_size[:2])
        # enc_output = self.rank_fc(rank_embedding(rank_emb, roi_feat_size[2])) + roi_feats
        enc_output = roi_feats

        # enc_output, _ = self.slf_attn(
        #     enc_output, enc_output, enc_output,
        #     attn_mask=rois_attn_mask, attn_pos_emb=rois_pos_emb)
        # enc_output = self.pos_ffn(enc_output)
        return enc_output