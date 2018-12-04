import argparse
parser = argparse.ArgumentParser(description="PyTorch code to train Structured Segment Networks (SSN)")
parser.add_argument('dataset', type=str, choices=['activitynet1.2', 'activitynet1.3', 'thumos14'])

parser.add_argument('--annotation_path', default='/data1/matheguo/important/data/activitynet/activity_net.v1-3.min_save.json',
                    type=str, help='Annotation file path')
parser.add_argument('--feat_root', default='/data1/matheguo/important/data/activitynet',
                    type=str, help='Feature directory path')
parser.add_argument('--result_path', default='/data1/matheguo/important/result/activitynet',
                    type=str, help='Result directory path')
parser.add_argument('--model', default='TAG', type=str,
                    help='(self_att | cluster')
parser.add_argument('--feat_model', default='i3d_rgb', type=str,
                    help='the model for extracting pretrained features ('
                    'i3d_rgb | i3d_rgb_trained | inception_resnet_v2 | inception_resnet_v2_trained)')
parser.add_argument('--use_flow', default=True, type=int,
                    help='whether use i3d_flow feature') # for self-attetion encoder
parser.add_argument('--pos_enc', default=False, type=int,
                    help='whether slice the original position indices of the input video sequence')
parser.add_argument('--att_kernel_type', default='self_attn',
                    type=str, help='the kernel type for attention computing, as in non-local networks (self_attn, concat, addition, dot, highorder)')
parser.add_argument('--n_layers', default=1,
                    type=int, help='the number of encoder layers in the self_attention encoder')
parser.add_argument('--reduce_dim', default=512,
                    type=int, help='if -1, not rediced; if > 0, reduce the input feature dimension first')
parser.add_argument('--n_head', default=8,
                    type=int, help='the number of attention head used in one encoder layer')
parser.add_argument('--d_inner_hid', default=2048, type=int,
                    help='the layer dimension for positionwise fc layers')
parser.add_argument('--prop_per_video', type=int, default=12)
parser.add_argument('--num_local', type=int, default=0)
parser.add_argument('--local_type', type=str, default='qkv')
parser.add_argument('--dilated_mask', type=int, default=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--groupwise_heads', type=int, default=0)
parser.add_argument('--roi_poolsize', type=str, default="1_3")

# ========================= Model Configs ==========================
parser.add_argument('--num_aug_segments', type=int, default=2)
parser.add_argument('--num_body_segments', type=int, default=5)

parser.add_argument('--dropout', '--do', default=0.1, type=float,
                    metavar='DO', help='dropout ratio (default: 0.8)')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--training_epoch_multiplier', '--tem', default=10, type=int,
                    help='replicate the training set by N times in one epoch')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-i', '--iter-size', default=1, type=int,
                    metavar='N', help='number of iterations before on update')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[3, 6], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-3)')
parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--comp_loss_weight', '--lw', default=0.1, type=float,
                    metavar='LW', help='the weight for the completeness loss')
parser.add_argument('--reg_loss_weight', '--rw', default=0.1, type=float,
                    metavar='LW', help='the weight for the location regression loss')

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')

# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')







