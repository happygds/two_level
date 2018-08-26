import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import torchvision.models
from attention.Layers import EncoderLayer, Local_EncoderLayer, ROI_Relation
from attention.proposal import proposal_layer
from attention.utils import *

class BinaryScore(torch.nn.Module):
    def __init__(self, args):
        super(BinaryScore, self).__init__()

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
        self.scores = nn.ModuleList([nn.Linear(args.d_model, 2)
                                     for _ in range(args.n_layers * len(self.multi_strides))])
        self.num_local = args.num_local
        self.dilated_mask = args.dilated_mask
        self.score_loss = None

    @property
    def loss(self):
        return self.score_loss

    def forward(self, feature, pos_ind, target=None, gts=None, feature_mask=None, test_mode=False):
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
        device_id = feature.device
        if not test_mode:
            rois, rois_mask, rois_relative_pos, labels = proposal_layer(score_outputs, gts=gts, test_mode=test_mode)
            # convert numpy to pytorch
            rois = torch.from_numpy(rois.astype('float32')).cuda().requires_grad_(False).to(device_id)
            rois_mask = torch.from_numpy(rois_mask.astype('float32')).cuda().requires_grad_(False).to(device_id)
            rois_relative_pos = torch.from_numpy(rois_relative_pos.astype('float32')).cuda().requires_grad_(False).to(device_id)
            labels = torch.from_numpy(labels.astype('float32')).cuda().requires_grad_(False).to(device_id)
            self.score_loss, self.attn_loss = self.build_loss(
                score_outputs, target, attns=enc_slf_attns, mask=feature_mask, multi_strides=self.multi_strides)
            return enc_input, rois, rois_mask, rois_relative_pos, labels
        else:
            rois, rois_mask, rois_relative_pos, actness = proposal_layer(score_outputs, test_mode=test_mode)
            rois = torch.from_numpy(rois.astype('float32')).cuda().requires_grad_(False).to(device_id)
            rois_mask = torch.from_numpy(rois_mask.astype('float32')).cuda().requires_grad_(False).to(device_id)
            rois_relative_pos = torch.from_numpy(rois_relative_pos.astype('float32')).cuda().requires_grad_(False).to(device_id)
            actness = torch.from_numpy(actness.astype('float32')).cuda().requires_grad_(False).to(device_id)
            return enc_output, rois, rois_mask, rois_relative_pos, actness


    def build_loss(self, inputs, target, attns=None, mask=None, multi_strides=None):
        targets = [target[:, (i//2)::i] for i in multi_strides]
        masks = [mask[:, (i//2)::i] for i in multi_strides]
        # targets = [target] * len(inputs)
        # masks = [mask] * len(inputs)

        weights = []
        for i, target in enumerate(targets):
            target = convert_categorical(target.cpu().numpy(), n_classes=2)
            target = torch.from_numpy(target).cuda().requires_grad_(False)
            target *= masks[i].unsqueeze(2)
            # cls_weight = 1. / target.mean(0).mean(0)
            weight = target.sum(1) / masks[i].sum(1).unsqueeze(1).clamp(eps)
            weight = 0.5 / weight.clamp(eps)
            # weight = weight / weight.mean(1).unsqueeze(1)
            targets[i] = target
            weights.append(weight)

        for i, x in enumerate(inputs):
            tmp_output = - targets[i] * torch.log(x.clamp(eps))
            tmp_output *= weights[i].unsqueeze(1)
            tmp_output = torch.sum(tmp_output.mean(2) * masks[i], dim=1) / \
                torch.sum(masks[i], dim=1).clamp(eps)
            tmp_output = torch.mean(tmp_output)
            if i == 0:
                output = tmp_output
            else:
                output += tmp_output
        output = output / len(inputs)

        if attns is not None:
            for i, (target, attn, mask) in enumerate(zip(targets, attns, masks)):
                # generate centered matrix
                tsize = target.size()
                H1, H2 = torch.eye(tsize[1], tsize[1]).unsqueeze(0).expand(tsize[0], -1, -1), \
                    (torch.ones((tsize[1], 1)) * torch.ones((1, tsize[1]))).unsqueeze(0).expand(tsize[0], -1, -1)
                H1, H2 = H1.cuda().requires_grad_(False), H2.cuda().requires_grad_(False)
                H = (H1 - H2 / target.sum(2, keepdim=True).sum(1, keepdim=True).clamp(eps)) * mask.unsqueeze(2) * mask.unsqueeze(1)
                target_cov = torch.bmm(target, target.transpose(1, 2))
                target_cov = torch.bmm(torch.bmm(H, target_cov), H) * mask.unsqueeze(2) * mask.unsqueeze(1)
                
                attn = attn.mean(1)
                attn = torch.bmm(torch.bmm(H, attn), H) * mask.unsqueeze(2) * mask.unsqueeze(1)
                tmp = torch.sqrt((attn * attn).sum(2).sum(1)) * torch.sqrt((target_cov * target_cov).sum(2).sum(1))
                tmp_output = 1. - (attn * target_cov).sum(2).sum(1).clamp(eps) / tmp.clamp(eps)
                tmp_output = (tmp_output * mask[:, 0]).mean()
                if i == 0:
                    attn_output = tmp_output
                else:
                    attn_output += tmp_output
            attn_output = attn_output / len(inputs)
    
        return output, attn_output


class BinaryClassifier(torch.nn.Module):
    def __init__(self, args):

        super(BinaryClassifier, self).__init__()

        self.rpn = BinaryScore(args)
        self.d_model = args.d_model

        self.roi_relations = ROI_Relation(args.d_model, args.roi_poolsize, args.d_inner_hid, 
                                          args.n_head, args.d_k, args.d_v, dropout=0.1)
        self.roi_cls = nn.Sequential(nn.Linear(args.d_model, 2), nn.Softmax(dim=2))
        self.roi_loss = None

    @property
    def loss(self):
        return self.roi_loss

    def forward(self, feature, pos_ind, target=None, gts=None, feature_mask=None, test_mode=False):
        if not test_mode:
            enc_output, rois, rois_mask, rois_relative_pos, labels = self.rpn(
                feature, pos_ind, target=target, gts=gts, feature_mask=feature_mask, test_mode=False)
        else:
            enc_output, rois, rois_mask, rois_relative_pos, actness = self.rpn(
                feature, pos_ind, target=target, gts=gts, feature_mask=feature_mask, test_mode=True)
        # use relative position embedding
        rois_pos_emb = pos_embedding(rois_relative_pos, self.d_model)
        roi_feats = self.roi_relations(enc_output, rois, rois_mask, rois_pos_emb)
        roi_scores = self.roi_cls(roi_feats)
        if not test_mode:
            self.roi_loss = self.build_loss(roi_scores, labels, rois_mask)
            return roi_scores
        return actness, roi_scores

    def build_loss(self, roi_scores, labels, rois_mask):
        labels *= rois_mask.unsqueeze(2)
        rois_weight = labels.sum(1) / rois_mask.sum(1).unsqueeze(1).clamp(eps)
        rois_weight = 0.5 / rois_weight.clamp(eps)

        rois_output = - labels * torch.log(roi_scores.clamp(eps))
        rois_output *= rois_weight.unsqueeze(1)
        rois_output = torch.sum(rois_output.mean(2) * rois_mask, dim=1) / \
            torch.sum(rois_mask, dim=1).clamp(eps)
        rois_output = torch.mean(rois_output)
    
        return rois_output
