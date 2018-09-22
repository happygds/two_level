import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

# from .torchsparseattn.fused import Fusedmax, FusedProxFunction
# from sparsemax import Sparsemax

eps = 1e-10

def convert_categorical(x_in, n_classes=2):
    shp = x_in.shape
    x = (x_in.ravel().astype('int'))
    x_mask = (x >= 0).reshape(-1, 1)
    x = x.clip(0)
    y = np.diag(np.ones((n_classes,)))
    y = y[x] * x_mask
    y = y.reshape(shp + (n_classes,)).astype('float32')
    return y

class CE_Criterion_multi(nn.Module):
    def __init__(self, use_weight=True, l_step=1.):
        super(CE_Criterion_multi, self).__init__()
        self.l_step = l_step
        self.use_weight = use_weight

    def forward(self, input, label, start, end, attn=None, mask=None):
        inputs = torch.split(input, 1, dim=2)
        targets = [label, start, end]

        if self.use_weight:
            weights = []
            for i, target in enumerate(targets):
                target = convert_categorical(target.cpu().numpy(), n_classes=2)
                target = torch.from_numpy(target).cuda().requires_grad_(False)
                target *= mask.unsqueeze(2)
                # cls_weight = 1. / target.mean(0).mean(0)
                weight = target.sum(1) / mask.sum(1).unsqueeze(1).clamp(eps)
                weight = 0.5 / weight.clamp(eps)
                # weight = weight / weight.mean(1).unsqueeze(1)
                targets[i] = target
                weights.append(weight)

        output = []
        for i, x in enumerate(inputs):
            x = torch.cat([1.-x, x], dim=2)
            tmp_output = - targets[i] * torch.log(x.clamp(eps))
            if self.use_weight:
                tmp_output *= weights[i].unsqueeze(1)
                tmp_output = torch.sum(tmp_output.mean(2) * mask, dim=1) / \
                    torch.sum(mask, dim=1).clamp(eps)
                tmp_output = torch.mean(tmp_output)
            output.append(tmp_output)
        score_loss, start_loss, end_loss = output

        if attn is not None:
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
            attn_output = (tmp_output * mask[:, 0]).mean() * self.l_step ** i
    
        return score_loss, start_loss, end_loss, attn_output


class CE_Criterion(nn.Module):
    def __init__(self, use_weight=True, iou_thres=[0.5, 0.6, 0.7, 0.8, 0.9]):
        super(CE_Criterion, self).__init__()
        self.use_weight = use_weight
        self.iou_thres = iou_thres

    def forward(self, x, y, mask):
        for i, iou_thre in enumerate(self.iou_thres):
            if self.use_weight:
                target = y[:, :, i, :]
                target *= mask.unsqueeze(2)
                # cls_weight = 1. / target.mean(0).mean(0)
                weight = target.sum(1) / mask.sum(1).unsqueeze(1).clamp(eps)
                weight = 0.5 / weight.clamp(eps)
                # weight = weight / weight.mean(1).unsqueeze(1)

            input = x[:, :, i, :]
            output = - target * torch.log(input.clamp(eps))
            if self.use_weight:
                output *= weight.unsqueeze(1)
                output = torch.sum(output.mean(2) * mask, dim=1) / \
                    torch.sum(mask, dim=1).clamp(eps)
                output = torch.mean(output)
            if i == 0:
                output_tmp = output
            else:
                output_tmp += output

        return output_tmp.mean() / len(self.iou_thres)


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

def pos_embedding(position_mat, feat_dim, wave_length=1000.):
    feat_range = torch.arange(0, feat_dim / 8)
    dim_mat = torch.pow(wave_length, (8. / feat_dim) * feat_range)
    dim_mat = dim_mat.view(1, 1, 1, 1, -1).cuda()
    pos_size = position_mat.size()
    position_mat = position_mat.unsqueeze(4)
    div_mat = torch.div(position_mat, dim_mat)
    sin_mat = torch.sin(div_mat)
    cos_mat = torch.cos(div_mat)
    embedding = torch.cat([sin_mat, cos_mat], dim=4)
    return embedding.view(pos_size[:3] + (feat_dim,)).float()

def rank_embedding(position_mat, feat_dim, wave_length=1000.):
    feat_range = torch.arange(0, feat_dim / 2)
    dim_mat = torch.pow(wave_length, (2. / feat_dim) * feat_range)
    dim_mat = dim_mat.view(1, 1, -1).cuda()
    pos_size = position_mat.size()
    position_mat = position_mat.unsqueeze(2)
    div_mat = torch.div(position_mat, dim_mat)
    sin_mat = torch.sin(div_mat)
    cos_mat = torch.cos(div_mat)
    embedding = torch.cat([sin_mat, cos_mat], dim=2)
    return embedding.view(pos_size[:2] + (feat_dim,)).float()

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

def get_attn_pos(attn_mask, num_local=16):
    ''' Get an attention with relative position embedding.'''
    attn_shape = attn_mask.size()
    xx, yy = np.mgrid[0:attn_shape[1], 0:attn_shape[2]]
    loc_ind = (yy - xx) % num_local
    mod_ind = (yy - xx) // num_local + num_local
    loc_ind, mod_ind = torch.from_numpy(loc_ind), torch.from_numpy(mod_ind)
    if attn_mask.is_cuda:
        loc_ind = loc_ind.cuda().float().requires_grad_(False)
        mod_ind = mod_ind.cuda().float().requires_grad_(False)
    return loc_ind, mod_ind        