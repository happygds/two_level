import torch
from torch import nn
import numpy as np

import torchvision.models
from attention import EncoderLayer, Local_EncoderLayer, Cluster_EncoderLayer


def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    row = np.arange(d_pos_vec) // 2 * 2 / d_pos_vec
    row = 1. / np.power(10000, row).reshape((1, -1))
    col = np.arange(n_position).reshape((-1, 1))
    position_enc = np.dot(col, row)

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

def score_embedding(position_mat, feat_dim, wave_length=1000):
    feat_range = torch.arange(0, feat_dim / 2)
    dim_mat = torch.pow(wave_length, (2. / feat_dim) * feat_range)
    dim_mat = dim_mat.view(1, 1, -1).cuda()
    position_mat = (100.0 * torch.log(position_mat / 0.5)).unsqueeze(2)
    div_mat = torch.div(position_mat, dim_mat)
    sin_mat = torch.sin(div_mat)
    cos_mat = torch.cos(div_mat)
    embedding = torch.cat([sin_mat, cos_mat], dim=2)
    return embedding

def get_attn_dilated_mask(attn_mask, num_local=16):
    ''' get the dilated mask to utilize the global information '''
    attn_shape = attn_mask.size()
    xx, yy = np.mgrid[0:attn_shape[1], 0:attn_shape[2]]
    dilated_mask = (np.abs(xx - yy) % num_local != 0).astype('uint8')
    dilated_ind = ((yy - xx) // num_local) * (1. - dilated_mask)
    dilated_ind = (dilated_ind + 512 // num_local) * (1. - dilated_mask)
    dilated_mask = torch.from_numpy(
        dilated_mask).unsqueeze(0).expand(attn_shape)
    dilated_ind = torch.from_numpy(dilated_ind)
    if attn_mask.is_cuda:
        dilated_mask = dilated_mask.cuda()
        # dilated_ind = dilated_ind.cuda().long()
    dilated_mask = torch.gt(attn_mask + dilated_mask, 0).requires_grad_(False)
    return dilated_mask


def get_attn_local_mask(attn_mask, num_local=16):
    ''' Get an attention mask for using only the local info.'''
    if num_local % 2 == 1:
        triu_k, tril_k = num_local // 2 + 1, num_local // 2 + 1
    else:
        triu_k, tril_k = num_local // 2, num_local // 2 + 1
    attn_shape = attn_mask.size()
    xx, yy = np.mgrid[0:attn_shape[1], 0:attn_shape[2]]
    local_mask = np.bitwise_or(xx - yy >= tril_k, yy - xx >= triu_k).astype('uint8')
    local_ind = ((yy - xx) + num_local // 2) * (1. - local_mask)
    local_mask = torch.from_numpy(local_mask).unsqueeze(0).expand(attn_shape)
    local_ind = torch.from_numpy(local_ind)
    if attn_mask.is_cuda:
        local_mask = local_mask.cuda()
        # local_ind = local_ind.cuda().long()
    local_mask = torch.gt(attn_mask + local_mask, 0).requires_grad_(False)
    return local_mask


class BinaryClassifier(torch.nn.Module):
    def __init__(self, num_class, course_segment, args, dropout=0.8, test_mode=False):

        super(BinaryClassifier, self).__init__()

        self.pos_enc = args.pos_enc
        self.reduce = args.reduce_dim > 0
        if self.reduce:
            self.reduce_layer = nn.Sequential(
                nn.Linear(args.input_dim, args.reduce_dim), nn.SELU())
        if args.dropout > 0:
            self.dropout = args.dropout
        else:
            self.dropout = 0.

        n_position, d_word_vec = 1200, args.d_model
        self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=0)
        self.position_enc.weight.data = position_encoding_init(
            n_position, d_word_vec)
        if args.num_local > 0:
            self.layer_stack = nn.ModuleList([
                Local_EncoderLayer(args.d_model, args.d_inner_hid, args.n_head, args.d_k,
                                   args.d_v, dropout=0.1, kernel_type=args.att_kernel_type, 
                                   local_type=args.local_type)
                for _ in range(args.n_layers)])
        elif args.n_cluster > 0:
            self.layer_stack = nn.ModuleList([
                Cluster_EncoderLayer(args.d_model, args.d_inner_hid, args.n_head, args.d_k,
                                     args.d_v, dropout=0.1, kernel_type=args.att_kernel_type, 
                                     n_cluster=args.n_cluster, local_type=args.local_type)
                for _ in range(args.n_layers)])
        else:
            self.layer_stack = nn.ModuleList([
                EncoderLayer(args.d_model, args.d_inner_hid, args.n_head, args.d_k,
                            args.d_v, dropout=0.1, kernel_type=args.att_kernel_type)
                for _ in range(args.n_layers)])

        self.d_model = args.d_model
        self.dropout = dropout
        self.test_mode = test_mode
        self.binary_classifier = nn.Linear(args.d_model, num_class)
        self.softmax = nn.Softmax(dim=-1)
        self.num_local = args.num_local
        self.dilated_mask = args.dilated_mask
        # self.layer_norm = nn.LayerNorm(args.d_model)
        self.score_att_layer = EncoderLayer(args.d_model, args.d_inner_hid, args.n_head, args.d_k,
                                            args.d_v, dropout=0.1, kernel_type='self_attn')
        self.binary_score = nn.Linear(args.d_model, num_class)

    def forward(self, feature, pos_ind, feature_mask=None, return_attns=False):
        # Word embedding look up
        if self.reduce:
            enc_input = self.reduce_layer(feature)
        else:
            enc_input = feature

        # # Position Encoding addition
        # if self.pos_enc:
        #     enc_input = enc_input + self.position_enc(pos_ind)
        # enc_input = self.layer_norm(enc_input)
        enc_slf_attns = []

        enc_output = enc_input
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

        for i, enc_layer in enumerate(self.layer_stack):
            enc_output, enc_slf_attn = enc_layer(
                enc_output, local_attn_mask=local_attn_mask, slf_attn_mask=enc_slf_attn_mask)
            enc_slf_attns += [enc_slf_attn]
        score_output = self.softmax(self.binary_classifier(enc_output))

        # use scores embedding
        score_embed = score_embedding(score_output[:, :, 1], self.d_model)
        print(score_embed[0, 0], score_embed[0, 10], score_embed[0, -1])
        import pdb
        pdb.set_trace()
        score_att_output, _ = self.score_att_layer(
                score_embed, slf_attn_mask=enc_slf_attn_mask)
        score_att_output = self.softmax(self.binary_score(score_att_output))

        return [score_output, score_att_output]

    def get_trainable_parameters(self):
        # ''' Avoid updating the position encoding '''
        enc_freezed_param_ids = set(map(id, self.position_enc.parameters()))
        freezed_param_ids = enc_freezed_param_ids
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)
