import argparse
import os
import math
import time
import shutil
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import pandas as pd
import numpy as np
from torch.nn.utils import clip_grad_norm_

from ssn_opts import parser
from load_binary_score import BinaryDataSet
from binary_model import BinaryClassifier
from transforms import *
from ops.utils import get_actionness_configs, ScheduledOptim
from ops.anet_db import ANetDB
from torch.utils import model_zoo
from attention.utils import Rank_Criterion, CE_Criterion_multi
from ops.AdamW import AdamW
from ops.eval_utils import area_under_curve, grd_activity
best_loss = 100


def convert_categorical(x_in, n_classes=2):
    shp = x_in.shape
    x = (x_in.ravel().astype('int'))
    x_mask = (x >= 0).reshape(-1, 1)
    x = x.clip(0)
    y = np.diag(np.ones((n_classes,)))
    y = y[x] * x_mask
    y = y.reshape(shp + (n_classes,)).astype('float32')
    return y


def main():
    global args, best_loss
    args = parser.parse_args()
    dataset_configs = get_actionness_configs(args.dataset)
    sampling_configs = dataset_configs['sampling']
    num_class = dataset_configs['num_class']
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    db = ANetDB.get_db("1.3")

    # set the directory for the rgb features
    if args.feat_model == 'i3d_rgb' or args.feat_model == 'i3d_rgb_trained':
        args.input_dim = 1024
    elif args.feat_model == 'inception_resnet_v2' or args.feat_model == 'inception_resnet_v2_trained':
        args.input_dim = 1536
    if args.use_flow:
        if not args.only_flow:
            args.input_dim += 1024
        else:
            args.input_dim = 1024
    print(("=> the input features are extracted from '{}' and the dim is '{}'").format(
        args.feat_model, args.input_dim))
    # if reduce the dimension of input feature first
    if args.reduce_dim > 0:
        assert args.reduce_dim % args.n_head == 0, "reduce_dim {} % n_head {} != 0".format(
            args.reduce_dim, args.n_head)
        args.d_k = int(args.reduce_dim // args.n_head)
        args.d_v = args.d_k
    else:
        assert args.input_dim % args.n_head == 0, "input_dim {} % n_head {} != 0".format(
            args.input_dim, args.n_head)
        args.d_k = int(args.input_dim // args.n_head)
        args.d_v = args.d_k
    args.d_model = args.n_head * args.d_k

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    if args.pos_enc:
        save_path = os.path.join(args.result_path, '_'.join(
            (args.att_kernel_type, 'N'+str(args.n_layers))))
    else:
        save_path = os.path.join(args.result_path, '_'.join(
            (args.att_kernel_type, 'N'+str(args.n_layers)))) + '_nopos'
    if args.num_local > 0:
        save_path = save_path + '_loc' + str(args.num_local) + args.local_type
        if args.dilated_mask:
            save_path += '_dilated'
    if args.groupwise_heads > 0:
        save_path = save_path + '_G' + str(args.groupwise_heads)
    if len(args.roi_poolsize) > 0:
        save_path = save_path + '_roi' + str(args.roi_poolsize)
    model_name = os.path.split(save_path)[1]
    # logger = Logger('./logs/{}'.format(model_name))
    logger = None

    model = BinaryClassifier(
        num_class, args.num_body_segments, args, dropout=args.dropout)
    model = torch.nn.DataParallel(model, device_ids=None).cuda()

    cudnn.enabled = False
    # cudnn.benchmark = True
    pin_memory = True
    train_prop_file = 'data/{}_proposal_list.txt'.format(
        dataset_configs['train_list'])
    val_prop_file = 'data/{}_proposal_list.txt'.format(
        dataset_configs['test_list'])
    train_videos = db.get_subset_videos('training')
    val_videos = db.get_subset_videos('validation')
    train_loader = torch.utils.data.DataLoader(
        BinaryDataSet(args.feat_root, args.feat_model, train_prop_file, train_videos,
                      exclude_empty=True, body_seg=args.num_body_segments,
                      input_dim=args.d_model, prop_per_video=args.prop_per_video,
                      fg_ratio=6, bg_ratio=6, num_local=args.num_local,
                      use_flow=args.use_flow, only_flow=args.only_flow),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=pin_memory,
        drop_last=True)

    # val_loader = torch.utils.data.DataLoader(
    #     BinaryDataSet(args.feat_root, args.feat_model, val_prop_file, val_videos,
    #                   exclude_empty=True, body_seg=args.num_body_segments,
    #                   input_dim=args.d_model, prop_per_video=args.prop_per_video,
    #                   fg_ratio=6, bg_ratio=6, num_local=args.num_local,
    #                   use_flow=args.use_flow, only_flow=args.only_flow),
    #     batch_size=args.batch_size//2, shuffle=False,
    #     num_workers=args.workers*2, pin_memory=pin_memory)
    val_loader = torch.utils.data.DataLoader(
        BinaryDataSet(args.feat_root, args.feat_model, val_prop_file, subset_videos=val_videos,
                      exclude_empty=True, body_seg=args.num_body_segments,
                      input_dim=args.d_model, test_mode=True, use_flow=args.use_flow,
                      verbose=False, num_local=args.num_local, only_flow=args.only_flow),
        batch_size=1, shuffle=False,
        num_workers=10, pin_memory=True)

    ground_truth, cls_to_idx = grd_activity('data/activity_net.v1-3.min_save.json', subset='validation')
    del cls_to_idx['background']

    # optimizer = torch.optim.Adam(
    #         model.parameters(),
    #         args.lr, weight_decay=args.weight_decay)

    optimizer = AdamW(
        model.parameters(),
        args.lr, weight_decay=args.weight_decay)

    # optimizer = torch.optim.SGD(model.parameters(),
    #                             args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay, nesterov=False)

    if args.resume is not None and len(args.resume) > 0:
        model.load_state_dict(torch.load(args.resume)[
                              'state_dict'], strict=False)
    criterion_stage1 = CE_Criterion_multi(use_weight=True)
    criterion_stage2 = Rank_Criterion(epsilon=0.02)

    patience = 0
    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch, args.lr_steps)
        # train for one epoch
        if patience > 5:
            break
        train(train_loader, model, optimizer, criterion_stage1,
              criterion_stage2, epoch, logger)

        # evaluate on validation list
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            loss = validate(
                val_loader, model, ground_truth, (epoch + 1) * len(train_loader), epoch)

        # remember best prec@1 and save checkpoint
            is_best = 1.0001 * loss < best_loss
            if is_best:
                patience = 0
            else:
                patience += 1
            best_loss = min(loss, best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.model,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
            }, is_best, save_path)


def train(train_loader, model, optimizer, criterion_stage1, criterion_stage2, epoch, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    score_losses = AverageMeter()
    start_losses = AverageMeter()
    end_losses = AverageMeter()
    roi_losses = AverageMeter()

    # switch to train model
    model.train()

    end_time = time.time()
    optimizer.zero_grad()

    for i, (feature, feature_mask, target, start, end, pos_ind, gts) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end_time)
        # feature_mask = feature.abs().mean(2).ne(0).float()
        feature = feature.cuda().requires_grad_(False)
        feature_mask = feature_mask.cuda().requires_grad_(False)
        pos_ind = pos_ind.cuda().requires_grad_(False)

        # compute output
        score_output, enc_slf_attn, roi_scores, labels, rois_mask = model(
            feature, pos_ind, target, gts=gts, feature_mask=feature_mask, epoch_id=epoch)
        score_loss, start_loss, end_loss, attn_loss = criterion_stage1(
            score_output, target, start, end, attn=enc_slf_attn, mask=feature_mask)
        roi_loss = criterion_stage2(roi_scores, labels, rois_mask)
        loss = score_loss + 20. * roi_loss + 0.5 * start_loss + 0.5 * end_loss
        score_losses.update(score_loss.item(), feature.size(0))
        start_losses.update(start_loss.item(), feature.size(0))
        end_losses.update(end_loss.item(), feature.size(0))
        roi_losses.update(roi_loss.item(), feature.size(0))
        losses.update(loss.item(), feature.size(0))

        # compute gradient and do SGD step
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(
                model.module.get_trainable_parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print('Clipping gradient: {} with coef {}'.format(
                    total_norm, args.clip_gradient / total_norm))
        else:
            total_norm = 0

        optimizer.step()

        # # 1. Log scalar values (scalar summary)
        # info = {'train_loss': loss.item(),
        #         'train_score_loss': score_loss.item(),
        #         'train_start_loss': start_loss.item(),
        #         'train_end_loss': end_loss.item(),
        #         'train_roi_loss': roi_loss.item()}
        # for tag, value in info.items():
        #     logger.scalar_summary(tag, value, i+epoch*len(train_loader)+1)
        # # 2. Log values and gradients of the parameters (histogram summary)
        # for tag, value in model.named_parameters():
        #     tag_ = tag.replace('.', '/')
        #     if np.isnan(value.grad.data.cpu().numpy()).any() or np.isnan(value.data.cpu().numpy()).any():
        #         import pdb; pdb.set_trace()
        #     logger.histo_summary(tag_, value.data.cpu().numpy(), i+epoch*len(train_loader)+1)
        #     logger.histo_summary(tag_+'/grad', value.grad.data.cpu().numpy(), i+epoch*len(train_loader)+1)

        del loss, score_loss, roi_loss, score_output, enc_slf_attn, roi_scores, labels, rois_mask
        # optimizer.update_learning_rate()
        optimizer.zero_grad()
        # measure elapsed time
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Score_Loss {score_loss.val:.4f} ({score_loss.avg:.4f})\t'
                  'Start_Loss {start_loss.val:.4f} ({start_loss.avg:.4f})\t'
                  'End_Loss {end_loss.val:.4f} ({end_loss.avg:.4f})\t'
                  'ROI_Loss {roi_loss.val:.4f} ({roi_loss.avg:.4f})\t'
                  .format(
                      epoch, i, len(train_loader), lr=optimizer.param_groups[0]['lr'],
                      batch_time=batch_time, data_time=data_time, loss=losses,
                      score_loss=score_losses, start_loss=start_losses,
                      end_loss=end_losses, roi_loss=roi_losses)
                  )


def validate(val_loader, model, ground_truth, iter, epoch):
    batch_time = AverageMeter()
    model.eval()
    end_time = time.time()

    video_lst, t_start_lst, t_end_lst, score_lst = [], [], [], []
    for i, (feature, feature_mask, num_feat, pos_ind, video_id, video_duration) in enumerate(val_loader):
        feature = feature[0].cuda()
        feature_mask = feature_mask[0].cuda()
        pos_ind = pos_ind[0].cuda()
        video_id = video_id[0]
        video_duration = float(video_duration[0].cpu().numpy())
        with torch.no_grad():
            rois, actness, roi_scores = model(
                feature, pos_ind, feature_mask=feature_mask, test_mode=True)
            rois, actness, roi_scores = rois[0].cpu().numpy(
            ), actness[0].cpu().numpy(), roi_scores[0].cpu().numpy()[:, 1]
            # import pdb; pdb.set_trace()
            rois = list(filter(lambda b: b[1]-b[0] > 0, rois))
            actness = list(filter(lambda b: b > 0, actness))
            roi_scores = list(filter(lambda b: b > 0, roi_scores))

            # save results
            video_lst.extend([video_id] * len(rois))
            t_start_lst.extend([x[0] / float(num_feat) * video_duration for x in rois])
            t_end_lst.extend([x[1] / float(num_feat) * video_duration for x in rois])
            score_lst.extend([roi_score*act_score for (act_score, roi_score) in zip(actness, roi_scores)])

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        if i % (1000) == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                  .format(i, len(val_loader), batch_time=batch_time))

    prediction = pd.DataFrame({'video-id': video_lst,
                                't-start': t_start_lst,
                                't-end': t_end_lst,
                                'score': score_lst})
    auc, ar_at_prop, nr_proposals_lst = area_under_curve(prediction, ground_truth, max_avg_nr_proposals=100,
                                                        tiou_thresholds=np.linspace(0.5, 0.95, 10))
    nr_proposals_lst = np.around(nr_proposals_lst)
    print('AR@1 is {:.6f}, AR@10 is {:.6f}, AR@20 is {:.6f}'.format(ar_at_prop[0], ar_at_prop[9], ar_at_prop[19]))
    print('AR@50 is {:.6f}, AR@100 is {:.6f}, AUC is {:.6f}'.format(ar_at_prop[49], ar_at_prop[99], auc))

    return -1. * auc


def save_checkpoint(state, is_best, save_path, filename='/checkpoint.pth.tar'):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = save_path + filename
    torch.save(state, filename)
    if is_best:
        best_name = save_path + '/model_best.pth.tar'
        shutil.copyfile(filename, best_name)


def adjust_learning_rate(optimizer, epoch, lr_steps):
    # Set the learning rate to the initial LR decayed by 10 every 30 epoches
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = decay


class AverageMeter(object):
    # Computes and stores the average and current value
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    target = target.view(-1)
    # computes the precision@k for the specific values of k
    maxk = max(topk)
    batch_size = target.size(0)
    output = output.view(batch_size, -1)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # print(output.size(), pred.size(), target.size())
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
