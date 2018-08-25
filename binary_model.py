import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import torchvision.models
from attention.Layers import EncoderLayer, Local_EncoderLayer, ROI_Relation
from attention.proposal import proposal_layer
from attention.utils import *

class BinaryClassifier(torch.nn.Module):
    def __init__(self, num_class, course_segment, args, dropout=0.8, test_mode=False):

        super(BinaryClassifier, self).__init__()

        self.reduce = args.reduce_dim > 0
        if self.reduce:
            self.reduce_layer = nn.Sequential(
                nn.Linear(args.input_dim, args.reduce_dim), nn.SELU())
        if args.dropout > 0:
            self.dropout = args.dropout
        else:
            self.dropout = 0.
        self.multi_strides = args.multi_strides
        self.n_layers = args.n_layers

        if args.num_local > 0:
            self.layer_stack = nn.ModuleList([
                Local_EncoderLayer(args.d_model, args.d_inner_hid, args.n_head, args.d_k,
                                   args.d_v, dropout=0.1, kernel_type=args.att_kernel_type, 
                                   local_type=args.local_type)
                for _ in range(args.n_layers * len(self.multi_strides))])
        else:
            self.layer_stack = nn.ModuleList([
                EncoderLayer(args.d_model, args.d_inner_hid, args.n_head, args.d_k,
                            args.d_v, dropout=0.1, kernel_type=args.att_kernel_type, 
                            groupwise_heads=args.groupwise_heads)
                for _ in range(args.n_layers * len(self.multi_strides))])

        self.d_model = args.d_model
        self.dropout = dropout
        self.test_mode = test_mode
        self.scores = nn.ModuleList([nn.Linear(args.d_model, 2)
                                     for _ in range(args.n_layers * len(self.multi_strides))])
        self.num_local = args.num_local
        self.dilated_mask = args.dilated_mask

        self.criterion_stage1 = CE_Criterion_multi()
        self.criterion_stage2 = CE_Criterion()
        self.roi_relations = ROI_Relation(args.d_model, args.roi_poolsize, args.d_inner_hid, 
                                           args.n_head, args.d_k, args.d_v, dropout=0.1)
        self.roi_cls = nn.Sequential(nn.Linear(args.d_model, 2), nn.Softmax(dim=2))

    def forward(self, feature, pos_ind, target=None, gts=None, feature_mask=None, test_mode=False):
        print(gts)
        import pdb; pdb.set_trace()
        # Word embedding look up
        if self.reduce:
            enc_input = self.reduce_layer(feature)
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

        score_outputs, enc_slf_attns = [], []
        size = enc_input.size()
        for scale, stride in enumerate(self.multi_strides[::-1]):
            layers, cls_layers = self.layer_stack[scale*self.n_layers:(scale+1)*self.n_layers], self.scores[scale*self.n_layers:(scale+1)*self.n_layers]
            if stride > 1:
                cur_output = F.pad(enc_input, (0, 0, 0, stride//2-1))
                cur_output = F.avg_pool1d(cur_output.transpose(1, 2), stride, 
                                          stride=stride).transpose(1, 2)
            else:
                cur_output = enc_input
            if scale == 0:
                enc_output = cur_output
            else:
                repeat = int(round(cur_output.size()[1] / enc_output.size()[1]))
                enc_output = F.upsample(enc_output.transpose(1, 2), scale_factor=repeat, 
                                        mode='nearest').transpose(1, 2)
                diff_size = cur_output.size()[1] - enc_output.size()[1]
                if diff_size != 0:
                    if diff_size > 0:
                        enc_output = F.pad(enc_output, (0, 0, 0, diff_size))
                    else:
                        enc_output = enc_output[:, :diff_size, :]
                enc_output += cur_output
            
            # obtain local and global mask
            slf_attn_mask = enc_slf_attn_mask[:, (stride//2)::stride, (stride//2)::stride]

            if local_attn_mask is not None:
                slf_local_mask = local_attn_mask[:, (stride//2)::stride, (stride//2)::stride]
            else:
                slf_local_mask = None

            for i, enc_layer in enumerate(layers):
                enc_output, enc_slf_attn = enc_layer(
                    enc_output, local_attn_mask=slf_local_mask, 
                    slf_attn_mask=slf_attn_mask)
            enc_slf_attns.append(enc_slf_attn)
            score_output = F.softmax(cls_layers[i](enc_output), dim=2)
            score_outputs.append(score_output)
        score_outputs = score_outputs[::-1]
        enc_slf_attns = enc_slf_attns[::-1]
        if test_mode:
            for scale, stride in enumerate(self.multi_strides):
                if scale > 0:
                    score_outputs[scale] = F.upsample(score_outputs[scale].transpose(1, 2), 
                                                      scale_factor=stride, mode='nearest').transpose(1, 2)
        # compute loss for training/validation stage
        if not test_mode:
            ce_loss, attn_loss = self.criterion_stage1(score_outputs, target, attns=enc_slf_attns, 
                                                       mask=feature_mask, multi_strides=self.multi_strides)
            rois, rois_mask, rois_relative_pos, labels = proposal_layer(score_outputs, gts=gts, test_mode=test_mode)
        else:
            rois, rois_mask, rois_relative_pos, actness = proposal_layer(score_outputs, test_mode=test_mode)

        # use relative position embedding
        rois_pos_emb = pos_embedding(rois_relative_pos, self.d_model)
        roi_feats = self.roi_relations(enc_output, rois, rois_mask, rois_pos_emb)
        roi_scores = self.roi_cls(roi_feats)
        if not test_mode:
            roi_loss = self.criterion_stage2(roi_scores, labels, rois_mask)
            return ce_loss, roi_loss

        return actness, roi_scores