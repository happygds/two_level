import torch
from torch import nn

from transforms import *
import torchvision.models


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


def get_attn_padding_mask(seq_q, seq_k):
    ''' Indicate the padding-related part to mask '''
    assert seq_q.dim() == 2 and seq_k.dim() == 2, \
        "seq_q.size() is {} and seq_k.size() is {}".format(seq_q.size(), seq_k.size())
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)   # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k)  # bxsqxsk
    return pad_attn_mask


class BinaryClassifier(torch.nn.Module):
    def __init__(self, num_class, course_segment, input_dim, dropout=0.8, test_mode=False):

        super(BinaryClassifier, self).__init__()

        self.pos_enc = opt.pos_enc
        self.reduce = opt.reduce_dim > 0
        if self.reduce:
            self.reduce_layer = nn.Sequential(
                nn.Linear(opt.input_dim, opt.reduce_dim), nn.SELU())
        if opt.dropout > 0:
            self.dropout = opt.dropout
        else:
            self.dropout = 0.

        n_position, d_word_vec = 1200, opt.d_model
        self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=0)
        self.position_enc.weight.data = position_encoding_init(
            n_position, d_word_vec)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(opt.d_model, opt.d_inner_hid, opt.n_head, opt.d_k,
                         opt.d_v, dropout=0.1, kernel_type=opt.att_kernel_type)
            for _ in range(opt.n_layers)])

        self.num_segments = course_segment
        self.course_segment = course_segment
        self.dropout = dropout
        self.test_mode = test_mode
        self.binary_classifier = nn.Linear(input_dim, num_class)
        self.softmax = nn.Softmax(dim=-1)



    def forward(self, feature, pos_ind, feature_mask=None, return_attns=False):
        # Word embedding look up
        if self.reduce:
            enc_input = self.reduce_layer(feature)
        else:
            enc_input = feature

        # Position Encoding addition
        if self.pos_enc:
            enc_input = enc_input + self.position_enc(pos_ind)
        enc_slf_attns = []

        enc_output = enc_input
        if feature_mask is not None:
            mb_size, len_k = enc_input.size()[:2]
            enc_slf_attn_mask = (
                1. - feature_mask).unsqueeze(1).expand(mb_size, len_k, len_k).byte()
        else:
            enc_slf_attn_mask = None
        for i, enc_layer in enumerate(self.layer_stack):
            enc_output, enc_slf_attn = enc_layer(
                enc_output, slf_attn_mask=enc_slf_attn_mask)
            enc_slf_attns += [enc_slf_attn]

        # enc_output = feature

        enc_output = self.actionness(enc_output)
        
        if return_attns:
            return enc_output, enc_slf_attns
        else:
            return enc_output,

    def get_trainable_parameters(self):
        # ''' Avoid updating the position encoding '''
        enc_freezed_param_ids = set(map(id, self.position_enc.parameters()))
        freezed_param_ids = enc_freezed_param_ids
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)

    # def forward(self, inputdata, target):
    #     if not self.test_mode:
    #         return self.train_forward(inputdata, target)
    #     else:
    #         return self.test_forward(inputdata)


    # def train_forward(self, inputdata, target):
        
    #     course_ft = inputdata[:, :, :, :].mean(dim=2)
    #     raw_course_ft = self.binary_classifier(course_ft).view((-1, 2))
    #     return raw_course_ft, target.view(-1)
                

    # def test_forward(self, inputdata):
    #     output = self.binary_classifier(inputdata)
    #     return self.softmax(output), inputdata
