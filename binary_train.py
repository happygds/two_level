import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from ssn_opts import parser
from load_binary_score import BinaryDataSet
from binary_model import BinaryClassifier
from transforms import *
from ops.utils import get_actionness_configs, ScheduledOptim
from ops.anet_db import ANetDB
from torch.utils import model_zoo
from attention.Modules import CE_Criterion
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
    args.dropout = 0.8
    torch.manual_seed(args.seed)
    db = ANetDB.get_db("1.3")

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

    train_videos = db.get_subset_videos('training')
    val_videos = db.get_subset_videos('validation')
    train_loader = torch.utils.data.DataLoader(
        BinaryDataSet(args.feat_root, args.feat_model, train_videos,
                      exclude_empty=True, body_seg=args.num_body_segments,
                      input_dim=args.d_model, prop_per_video=args.prop_per_video,
                      fg_ratio=6, bg_ratio=6, num_local=args.num_local, use_flow=args.use_flow),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=pin_memory,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        BinaryDataSet(args.feat_root, args.feat_model, val_videos,
                      exclude_empty=True, body_seg=args.num_body_segments,
                      input_dim=args.d_model, prop_per_video=args.prop_per_video,
                      fg_ratio=6, bg_ratio=6, num_local=args.num_local, use_flow=args.use_flow),
        batch_size=args.batch_size * 3 // 2, shuffle=False,
        num_workers=args.workers, pin_memory=pin_memory)

    binary_criterion = CE_Criterion(use_weight=True)

    optimizer = torch.optim.Adam(
            model.module.get_trainable_parameters(),
            args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.module.get_trainable_parameters(),
    #                             args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay, nesterov=False)

    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch, args.lr_steps)
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

    # switch to train model
    model.train()

    end = time.time()
    optimizer.zero_grad()

    for i, (feature, target, pos_ind) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        feature_mask = feature.abs().mean(2).ne(0).float()
        feature = feature.cuda().requires_grad_(False)
        feature_mask = feature_mask.cuda().requires_grad_(False)
        pos_ind = pos_ind.cuda().requires_grad_(False)

        target = convert_categorical(target.cpu().numpy(), n_classes=2)
        target = torch.from_numpy(target).cuda().requires_grad_(False)
        target *= feature_mask.unsqueeze(2)
        # cls_weight = 1. / target.mean(0).mean(0)
        cls_weight = (feature_mask.sum(1).unsqueeze(1) / target.sum(1)).mean(0)

        # compute output
        binary_score = model(feature, pos_ind, feature_mask=feature_mask)

        loss = criterion(binary_score, target, weight=cls_weight, mask=feature_mask)
        losses.update(loss.item(), feature.size(0))

        # compute gradient and do SGD step
        loss.backward()

        if i % args.iter_size == 0:
            # scale down gradients when iter size is functioning
            if args.iter_size != 1:
                for g in optimizer.param_groups:
                    for p in g['params']:
                        p.grad /= args.iter_size

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(
                model.module.get_trainable_parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print('Clipping gradient: {} with coef {}'.format(
                    total_norm, args.clip_gradient / total_norm))
        else:
            total_norm = 0

        optimizer.step()
        # optimizer.update_learning_rate()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, lr=optimizer.param_groups[0]['lr'])
                  )


def validate(val_loader, model, criterion, iter):
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.eval()

    end = time.time()

    for i, (feature, target, pos_ind) in enumerate(val_loader):
        with torch.no_grad():
            feature_mask = feature.abs().mean(2).ne(0).float()
            feature = feature.cuda().requires_grad_(False)
            feature_mask = feature_mask.cuda().requires_grad_(False)
            pos_ind = pos_ind.cuda().requires_grad_(False)

            target = convert_categorical(target.cpu().numpy(), n_classes=2)
            target = torch.from_numpy(target).cuda().requires_grad_(False)
            target *= feature_mask.unsqueeze(2)
            # cls_weight = 1. / target.mean(0).mean(0)
            cls_weight = (feature_mask.sum(1).unsqueeze(1) / target.sum(1)).mean(0)

            # compute output
            binary_score = model(feature, pos_ind, feature_mask=feature_mask)

            loss = criterion(binary_score, target, weight=cls_weight, mask=feature_mask)
        losses.update(loss.item(), feature.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.4f} ({loss.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses))

    print('Testing Results: Loss {loss.avg:.5f} \t'
          .format(loss=losses))

    return losses.avg


def save_checkpoint(state, is_best, filename='/checkpoint.pth.tar'):
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
    if args.n_cluster > 0:
        save_path = save_path + '_C' + str(args.n_cluster)
        
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
