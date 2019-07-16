import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

from attention.Layers import EncoderLayer, Local_EncoderLayer, ROI_Relation
from attention.proposal import proposal_layer
from attention.utils import *


class BinaryClassifier(torch.nn.Module):
    def __init__(self, num_class, course_segment, args, dropout=0.5, test_mode=False):

        super(BinaryClassifier, self).__init__()

        if args.dropout > 0:
            self.dropout = args.dropout
        else:
            self.dropout = 0.
        self.reduce = args.reduce_dim > 0
        if self.reduce:
            self.reduce_layer = nn.Sequential(
                nn.Linear(args.input_dim, args.reduce_dim), nn.SELU(), nn.Dropout(self.dropout))
            # self.inputnorm = nn.BatchNorm1d(args.d_model)
        self.n_layers = args.n_layers

        self.layer_stack = nn.Sequential(nn.Conv1d(args.d_model, args.d_model, 3, padding=1),
                                         nn.ReLU(), nn.Dropout(self.dropout), 
                                         nn.Conv1d(args.d_model,args.d_model, 3, padding=1),
                                         nn.ReLU(), nn.Dropout(self.dropout))

        self.d_model = args.d_model
        self.test_mode = test_mode
        self.scores = nn.Linear(args.d_model, 3)
        self.num_local = args.num_local
        self.dilated_mask = args.dilated_mask
        self.trn_kernel = args.groupwise_heads

        self.roi_relations = ROI_Relation(args.d_model, args.roi_poolsize, args.d_inner_hid,
                                          args.n_head, args.d_k, args.d_v, dropout=self.dropout)
        self.norm = nn.BatchNorm1d(args.d_model)
        # self.roi_feat_max = nn.Sequential(
        #     nn.Linear(args.d_model, args.d_model), nn.SELU(), nn.Dropout(self.dropout))
        # self.w_roi = nn.Parameter(torch.FloatTensor(1, 1, args.d_model))
        # init.xavier_normal_(self.w_roi)
        self.roi_cls = nn.Linear(args.d_model, 2)

    def forward(self, feature, pos_ind, target=None, gts=None,
                feature_mask=None, test_mode=False, epoch_id=None):
        # Word embedding look up
        if self.reduce:
            enc_input = self.reduce_layer(feature)
            # enc_input = self.inputnorm(enc_input.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        else:
            enc_input = feature

        enc_output = enc_input
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
        roi_feats = self.roi_relations(
            enc_input, start_rois, end_rois, rois, rois_mask, rois_pos_emb)
        roi_feats = self.norm(roi_feats.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        # roi_feat_max = self.roi_feat_max(roi_feats).max(1)[0].unsqueeze(1)
        # roi_feat_max = self.w_roi
        roi_scores = F.softmax(self.roi_cls(roi_feats), dim=2)
        # roi_scores = ((roi_feat_max * roi_feats).sum(2) / torch.sqrt(
        #     (roi_feat_max ** 2).sum(2) * (roi_feats ** 2).sum(2)).clamp(1e-14)).clamp(0.)

        if not test_mode:
            return score_output, enc_slf_attn, roi_scores, labels, rois_mask

        return rois[:, :, 1:], actness, roi_scores
