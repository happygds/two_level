''' Define the Layers '''
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
# from .sparsemax import Sparsemax
from .Modules import ScaledDotProductAttention, MultiHeadAttention, PositionwiseFeedForward
from .utils import rank_embedding, roi_embedding
from broi1d_pooling_avg.modules.broi1d_pool import BRoI1DPool
from broi1d_align.broi1d_align import BRoI1DAlign


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v,
                 dropout=0.5, kernel_type='self_attn', groupwise_heads=0):
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
                 d_k, d_v, dropout=0.5, kernel_type='roi_remov'):
        super(ROI_Relation, self).__init__()
        if len(roipool_size) == 1:
            start_pool_size = 1
            roipool_size = int(roipool_size)
        else:
            assert len(roipool_size) == 3 and roipool_size[1] == '_'
            start_pool_size, roipool_size = map(int, roipool_size.split("_"))
        print("start_pool_size is {} and inner_pool_size is {}".format(
            start_pool_size, roipool_size))
        self.roi_pool = BRoI1DPool(
            roipool_size, 1., start_pool_size, start_pool_size, 1./5)
        # self.roi_pool = BRoI1DAlign(roipool_size, 1., start_pool_size, start_pool_size, 1./5)
        self.bpool_size = start_pool_size
        self.roipool_size = roipool_size
        # self.roi_conv = nn.Sequential(nn.Conv1d(d_model, d_model, 3, padding=1), nn.Dropout(dropout), nn.SELU(),
        #                               nn.Conv1d(d_model, d_model, 3, padding=1), nn.Dropout(dropout), nn.SELU(),
        #                               nn.AvgPool1d(5, stride=4, padding=2))
        # in_ch = ((2*self.bpool_size+self.roipool_size) - 1) // 4 + 1
        # self.roi_fc = nn.Sequential(nn.Linear(d_model*in_ch, d_model), nn.Dropout(dropout), nn.SELU())

        self.left_fc = nn.Sequential(nn.Linear(
            (self.bpool_size+roipool_size//2+1)*d_model, d_model), nn.Dropout(dropout), nn.SELU())
        self.inner_fc = nn.Sequential(
            nn.Linear(roipool_size*d_model, d_model), nn.Dropout(dropout), nn.SELU())
        self.right_fc = nn.Sequential(nn.Linear(
            (self.bpool_size+roipool_size//2+1)*d_model, d_model), nn.Dropout(dropout), nn.SELU())
        self.roi_fc = nn.Sequential(
            nn.Linear(2*d_model, d_model), nn.Dropout(dropout), nn.SELU())

    def forward(self, features, start_rois, end_rois, rois, rois_mask, rois_pos_emb):
        len_feat = features.size()[1]
        features = features.transpose(1, 2)
        roi_feats = self.roi_pool(features, rois)
        roi_feat_size = roi_feats.size()
        # roi_feats = self.roi_conv(roi_feats.view((-1,)+roi_feat_size[2:])).view(roi_feat_size[:2]+(-1,))
        # roi_feats = self.roi_fc(roi_feats).view(roi_feat_size[:3])

        # use SSN-like fc-layers
        left_feats = self.left_fc(roi_feats[:, :, :, 0:(
            self.roipool_size//2+self.bpool_size+1)].contiguous().view(roi_feat_size[:2]+(-1,)))
        inner_feats = self.inner_fc(roi_feats[:, :, :, self.bpool_size:(
            self.roipool_size+self.bpool_size)].contiguous().view(roi_feat_size[:2]+(-1,)))
        right_feats = self.left_fc(roi_feats[:, :, :, (self.roipool_size//2+self.bpool_size):(
            self.roipool_size+2*self.bpool_size)].contiguous().view(roi_feat_size[:2]+(-1,)))
        roi_feats = self.roi_fc(torch.cat([right_feats - left_feats, inner_feats], dim=2))

        # compute mask
        mb_size, len_k = roi_feats.size()[:2]
        rois_attn_mask = (
            1. - rois_mask).unsqueeze(1).expand(mb_size, len_k, len_k).byte()
        rois_attn_mask = torch.gt(
            rois_attn_mask + rois_attn_mask.transpose(1, 2), 0)
        # use rank embedding
        # rank_emb = torch.arange(roi_feat_size[1]).view((1, -1)).float().cuda().requires_grad_(False).expand(roi_feat_size[:2]) + 1
        # enc_output = self.rank_fc(rank_embedding(rank_emb, roi_feat_size[2])) + roi_feats

        # rois_cent, rois_dura = rois[:, :, 1:].mean(2).unsqueeze(2), (rois[:, :, 2] - rois[:, :, 1]).unsqueeze(2)
        # rois_emb = torch.cat([rois_cent, rois_dura], dim=2)
        # rois_emb = 20. * torch.log((rois_emb / (len_feat * 0.5)).clamp(1e-3))
        # import pdb; pdb.set_trace()
        # enc_output = roi_feats + F.selu(self.rois_emb(roi_embedding(rois[:, :, 1:], roi_feat_size[2])))
        enc_output = roi_feats

        return enc_output
