import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from attention.Layers import EncoderLayer, Local_EncoderLayer, ROI_Relation
from attention.proposal import proposal_layer
from attention.utils import *

class BinaryClassifier(torch.nn.Module):
    def __init__(self, num_class, course_segment, args, dropout=0.1, test_mode=False):
    
        super(BinaryClassifier, self).__init__()

        self.reduce = args.reduce_dim > 0
        if self.reduce:
            self.reduce_layer = nn.Sequential(
                nn.Linear(args.input_dim, args.reduce_dim), nn.SELU())
        if args.dropout > 0:
            self.dropout = args.dropout
        else:
            self.dropout = 0.
        self.n_layers = args.n_layers

        if args.num_local > 0:
            self.layer_stack = nn.ModuleList([
                Local_EncoderLayer(args.d_model, args.d_inner_hid, args.n_head, args.d_k,
                                   args.d_v, dropout=0.1, kernel_type=args.att_kernel_type, 
                                   local_type=args.local_type)
                for _ in range(args.n_layers)])
        else:
            self.layer_stack = nn.ModuleList([
                EncoderLayer(args.d_model, args.d_inner_hid, args.n_head, args.d_k,
                            args.d_v, dropout=0.1, kernel_type=args.att_kernel_type, 
                            groupwise_heads=args.groupwise_heads)
                for _ in range(args.n_layers)])

        self.d_model = args.d_model
        self.dropout = dropout
        self.test_mode = test_mode
        self.scores = nn.Linear(args.d_model, 3)
        self.num_local = args.num_local
        self.dilated_mask = args.dilated_mask
        self.trn_kernel = args.groupwise_heads

        self.roi_relations = ROI_Relation(args.d_model, args.roi_poolsize, args.d_inner_hid, 
                                          args.n_head, args.d_k, args.d_v, dropout=0.1)
        self.batchnorm = nn.BatchNorm1d(args.d_model)
        self.roi_cls = nn.Linear(args.d_model, 2)

    def forward(self, feature, pos_ind, target=None, gts=None, 
                feature_mask=None, test_mode=False, epoch_id=None):
        # Word embedding look up
        if self.reduce:
            enc_input = self.reduce_layer(feature)
            enc_input = self.batchnorm(enc_input.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        else:
            enc_input = feature

        mb_size, len_k = enc_input.size()[:2]
        if feature_mask is not None:
            enc_slf_attn_mask = (
                1. - feature_mask).unsqueeze(1).expand(mb_size, len_k, len_k).byte()
        else:
            enc_slf_attn_mask = torch.zeros((mb_size, len_k, len_k)).byte().cuda()
        local_attn_mask = None
        if self.num_local > 0:
            local_attn_mask = get_attn_local_mask(enc_slf_attn_mask, num_local=self.num_local)
            if self.dilated_mask:
                enc_slf_attn_mask = get_attn_dilated_mask(enc_slf_attn_mask, num_local=self.num_local)
        enc_slf_attn_mask = torch.gt(enc_slf_attn_mask + enc_slf_attn_mask.transpose(1, 2), 0)

        size = enc_input.size()
        enc_output = enc_input
        # obtain local and global mask
        slf_attn_mask = enc_slf_attn_mask
        if local_attn_mask is not None:
            slf_local_mask = local_attn_mask
        else:
            slf_local_mask = None

        for i, enc_layer in enumerate(self.layer_stack):
            enc_output, enc_slf_attn = enc_layer(
                enc_output, local_attn_mask=slf_local_mask, 
                slf_attn_mask=slf_attn_mask)
        score_output = F.sigmoid(self.scores(enc_output))

        # compute loss for training/validation stage
        if not test_mode:
            start_rois, end_rois, rois, rois_mask, rois_relative_pos, labels = proposal_layer(
                score_output, feature_mask, gts=gts, test_mode=test_mode, epoch_id=epoch_id)
        else:
            start_rois, end_rois, rois, rois_mask, rois_relative_pos, actness = proposal_layer(
                score_output, feature_mask, test_mode=test_mode)

        # use relative position embedding
        rois_pos_emb = pos_embedding(rois_relative_pos, self.d_model)
        roi_feats = self.roi_relations(enc_input, start_rois, end_rois, rois, rois_mask, rois_pos_emb)
        roi_scores = F.softmax(self.roi_cls(roi_feats), dim=2)
        # import pdb; pdb.set_trace()

        if not test_mode:
            return score_output, enc_slf_attn, roi_scores, labels, rois_mask

        return rois[:, :, 1:], actness, roi_scores