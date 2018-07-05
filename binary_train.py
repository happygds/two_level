import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm

from ssn_opts import parser
from load_binary_score import BinaryDataSet
from binary_model import BinaryClassifier
from transforms import *
from ops.utils import get_actionness_configs
from torch.utils import model_zoo
best_loss = 100


def main():
    global args, best_loss
    args = parser.parse_args()
    dataset_configs = get_actionness_configs(args.dataset)
    sampling_configs = dataset_configs['sampling']
    num_class = dataset_configs['num_class']
    args.dropout = 0.8

    # set the directory for the rgb features
    if args.feat_model == 'i3d_rgb' or args.feat_model == 'i3d_rgb_trained':
        args.input_dim = 1024
    elif args.feat_model == 'inception_resnet_v2' or args.feat_model == 'inception_resnet_v2_trained':
        args.input_dim = 1536
    if args.use_flow:
        args.input_dim += 1024
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

    model = BinaryClassifier(
        num_class, args.num_body_segments, args, dropout=args.dropout)
    model = torch.nn.DataParallel(model, device_ids=None).cuda()

    cudnn.benchmark = True
    pin_memory = True

    train_prop_file = 'data/{}_proposal_list.txt'.format(
        dataset_configs['train_list'])
    val_prop_file = 'data/{}_proposal_list.txt'.format(
        dataset_configs['test_list'])
    train_loader = torch.utils.data.DataLoader(
        BinaryDataSet(args.feat_root, args.feat_model, train_prop_file,
                      exclude_empty=True, body_seg=args.num_body_segments,
                      input_dim=args.d_model, prop_per_video=args.prop_per_video,
                      fg_ratio=6, bg_ratio=6),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=pin_memory,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        BinaryDataSet(args.feat_root, args.feat_model, val_prop_file,
                      exclude_empty=True, body_seg=args.num_body_segments,
                      input_dim=args.d_model, prop_per_video=args.prop_per_video,
                      fg_ratio=6, bg_ratio=6),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=pin_memory)

    binary_criterion = torch.nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.module.get_trainable_parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)
        # train for one epoch
        train(train_loader, model, binary_criterion, optimizer, epoch)

        # evaluate on validation list
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            loss = validate(val_loader, model, binary_criterion,
                            (epoch + 1) * len(train_loader))

        # remember best prec@1 and save checkpoint
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.model,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    fg_accuracies = AverageMeter()
    bg_accuracies = AverageMeter()

    # switch to train model
    model.train()

    end = time.time()
    optimizer.zero_grad()

    for i, (feature, pos_ind, sel_prop_inds, prop_type_target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        feature_mask = feature.abs().mean(2).ne(0).float()
        feature = torch.autograd.Variable(feature, requires_grad=False).cuda()
        feature_mask = torch.autograd.Variable(feature_mask, requires_grad=False).cuda()
        pos_ind = torch.autograd.Variable(pos_ind, requires_grad=False).cuda()
        sel_prop_inds = torch.autograd.Variable(sel_prop_inds, requires_grad=False).cuda()
        prop_type_target = torch.autograd.Variable(prop_type_target, requires_grad=False).cuda()

        # compute output
        binary_score, prop_type_target = model(
            feature, pos_ind, sel_prop_ind=sel_prop_inds, feature_mask=feature_mask, target=prop_type_target)

        # print(binary_score.size(), prop_type_target.size())
        print(prop_type_target, sel_prop_inds)
        loss = criterion(binary_score, prop_type_target)

        losses.update(loss.item(), feature.size(0))
        fg_num_prop = args.prop_per_video//2*args.num_body_segments
        fg_acc = accuracy(binary_score.view(-1, 2, fg_num_prop, binary_score.size(1))[:, 0, :, :].contiguous(),
                        prop_type_target.view(-1, 2, fg_num_prop)[:, 0, :].contiguous())
        bg_acc = accuracy(binary_score.view(-1, 2, fg_num_prop, binary_score.size(1))[:, 1, :, :].contiguous(),
                        prop_type_target.view(-1, 2, fg_num_prop)[:, 1, :].contiguous())

        fg_accuracies.update(fg_acc[0].item(), binary_score.size(0) // 2)
        bg_accuracies.update(bg_acc[0].item(), binary_score.size(0) // 2)

        # compute gradient and do SGD step
        loss.backward()

        if i % args.iter_size == 0:
            # scale down gradients when iter size is functioning
            if args.iter_size != 1:
                for g in optimizer.param_groups:
                    for p in g['params']:
                        p.grad /= args.iter_size

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(model.module.get_trainable_parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print('Clipping gradient: {} with coef {}'.format(
                    total_norm, args.clip_gradient / total_norm))
        else:
            total_norm = 0

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                '\n FG{fg_acc.val:.02f}({fg_acc.avg:.02f}) BG {bg_acc.val:.02f} ({bg_acc.avg:.02f})'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, lr=optimizer.param_groups[0]['lr'],
                    fg_acc=fg_accuracies, bg_acc=bg_accuracies)
                )


def validate(val_loader, model, criterion, iter):
    batch_time = AverageMeter()
    losses = AverageMeter()
    fg_accuracies = AverageMeter()
    bg_accuracies = AverageMeter()

    model.eval()

    end = time.time()

    for i, (feature, pos_ind, sel_prop_inds, prop_type_target) in enumerate(val_loader):
        with torch.no_grad():
            feature_mask = feature.abs().mean(2).ne(0).float()
            feature = torch.autograd.Variable(feature).cuda()
            feature_mask = torch.autograd.Variable(feature_mask).cuda()
            pos_ind = torch.autograd.Variable(pos_ind).cuda()
            sel_prop_inds = torch.autograd.Variable(sel_prop_inds).cuda()
            prop_type_target = torch.autograd.Variable(prop_type_target).cuda()
            # compute output
            binary_score, prop_type_target = model(feature, pos_ind, sel_prop_ind=sel_prop_inds,
                                 feature_mask=feature_mask, target=prop_type_target)

        loss = criterion(binary_score, prop_type_target)
        losses.update(loss.item(), feature.size(0))
        fg_num_prop = args.prop_per_video//2*args.num_body_segments
        fg_acc = accuracy(binary_score.view(-1, 2, fg_num_prop, binary_score.size(1))[:, 0, :, :].contiguous(),
                          prop_type_target.view(-1, 2, fg_num_prop)[:, 0, :].contiguous())
        bg_acc = accuracy(binary_score.view(-1, 2, fg_num_prop, binary_score.size(1))[:, 1, :, :].contiguous(),
                          prop_type_target.view(-1, 2, fg_num_prop)[:, 1, :].contiguous())

        fg_accuracies.update(fg_acc[0].item(), binary_score.size(0) // 2)
        bg_accuracies.update(bg_acc[0].item(), binary_score.size(0) // 2)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.4f} ({loss.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'FG {fg_acc.val:.02f} BG {bg_acc.val:.02f}'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      fg_acc=fg_accuracies, bg_acc=bg_accuracies))

    print('Testing Results: Loss {loss.avg:.5f} \t'
          'FG Acc. {fg_acc.avg:.02f} BG Acc. {bg_acc.avg:.02f}'
          .format(loss=losses, fg_acc=fg_accuracies, bg_acc=bg_accuracies))

    return losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = args.result_path + '_'.join((args.att_kernel_type, args.n_layers, filename))
    torch.save(state, filename)
    if is_best:
        best_name = args.result_path + '/model_best.pth.tar'
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
