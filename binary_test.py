import argparse
import time
import pdb
import numpy as np

from load_binary_score import BinaryDataSet
from binary_model import BinaryClassifier
from transforms import *

from torch import multiprocessing
from torch.utils import model_zoo
from torch.autograd import Variable
from ops.anet_db import ANetDB
from ops.thumos_db import THUMOSDB
from ops.utils import get_actionness_configs, get_reference_model_url

global args
parser = argparse.ArgumentParser(description='extract actionnes score')
parser.add_argument('dataset', type=str, choices=[
                    'activitynet1.2', 'activitynet1.3', 'thumos14'])
parser.add_argument('subset', type=str, choices=[
                    'training', 'validation', 'testing'])
parser.add_argument('weights', type=str)
parser.add_argument('save_scores', type=str)
parser.add_argument('--save_raw_scores', type=str, default=None)
parser.add_argument('--frame_interval', type=int, default=5)
parser.add_argument('--test_batchsize', type=int, default=32)
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--num_body_segments', type=int, default=5)
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--annotation_path', default='../THUMOS14/thumos_annots.json',
                    type=str, help='Annotation file path')
parser.add_argument('--feat_root', default='../THUMOS14',
                    type=str, help='Feature directory path')
parser.add_argument('--result_path', default='../THUMOS14/result_two_level',
                    type=str, help='Result directory path')
parser.add_argument('--model', default='TAG', type=str,
                    help='(self_att | TAG')
parser.add_argument('--feat_model', default='feature_anet_200', type=str,
                    help='the model for extracting pretrained features ('
                    'feature_anet_200 | c3d_feature)')
parser.add_argument('--use_flow', default=True, type=int,
                    help='whether use i3d_flow feature')
parser.add_argument('--only_flow', default=False, type=int,
                    help='whether only use i3d_flow feature') # for self-attetion encoder
parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
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
parser.add_argument('--groupwise_heads', type=int, default=0)
parser.add_argument('--roi_poolsize', type=str, default="1_3")

args = parser.parse_args()

dataset_configs = get_actionness_configs(args.dataset)
num_class = dataset_configs['num_class']

if args.dataset == 'thumos14':
    if args.subset == 'validation':
        test_prop_file = 'data/{}_proposal_list.txt'.format(
            dataset_configs['train_list'])
    elif args.subset == 'testing':
        test_prop_file = 'data/{}_proposal_list.txt'.format(
            dataset_configs['test_list'])
elif args.dataset == 'activitynet1.2':
    if args.subset == 'training':
        test_prop_file = 'data/{}_proposal_list.txt'.format(
            dataset_configs['train_list'])
    elif args.subset == 'validation':
        test_prop_file = 'data/{}_proposal_list.txt'.format(
            dataset_configs['test_list'])
elif args.dataset == 'activitynet1.3':
    if args.subset == 'training':
        test_prop_file = 'data/{}_proposal_list.txt'.format(
            dataset_configs['train_list'])
    elif args.subset == 'validation':
        test_prop_file = 'data/{}_proposal_list.txt'.format(
            dataset_configs['test_list'])
    else:
        test_prop_file = None

# set the directory for the rgb features
if args.feat_model == 'feature_anet_200':
    args.input_dim = 200
elif args.feat_model == 'c3d_feature':
    args.input_dim = 487
    assert args.use_flow is not True
if args.use_flow:
    if not args.only_flow:
        args.input_dim *= 2
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

gpu_list = args.gpus if args.gpus is not None else range(1)


def runner_func(dataset, state_dict, gpu_id, index_queue, result_queue):
    torch.cuda.set_device(gpu_id)
    net = BinaryClassifier(num_class, args.num_body_segments,
                           args, dropout=args.dropout, test_mode=True)
    # net = torch.nn.DataParallel(net, device_ids=[gpu_id])

    net.load_state_dict(state_dict)
    net.eval()
    net.cuda()
    while True:
        index = index_queue.get()
        feats_gen, video_id = dataset[index]

        rois, roi_scores = [], []
        for feat_gen in feats_gen:
            feature, feature_mask, seg_ind, pos_ind = feat_gen
            feature = feature.cuda()
            feature_mask = feature_mask.cuda()
            pos_ind = pos_ind.cuda()
            video_id = video_id
            with torch.no_grad():
                this_rois, this_actness, this_roi_scores = net(
                    feature, pos_ind, feature_mask=feature_mask, test_mode=True)
                this_rois, this_actness, this_roi_scores = this_rois.cpu().numpy(
                ), this_actness.cpu().numpy(), this_roi_scores.cpu().numpy()[:, :, 1]
                this_rois += seg_ind.cpu().numpy().reshape((-1, 1, 1))
                this_roi_scores *= this_actness
                
                this_rois, this_roi_scores = this_rois.reshape((-1, 2)), this_roi_scores.reshape((-1))
                if len(rois) == 0:
                    rois, roi_scores = this_rois, this_roi_scores
                else:
                    rois = np.concatenate((rois, this_rois), axis=0)
                    roi_scores = np.concatenate((roi_scores, this_roi_scores), axis=0)
        # print("the number of rois is {}".format(len(rois)))
        outputs = [list(rois), list(roi_scores)]

        result_queue.put(
            (dataset.video_list[index].id.split('/')[-1], outputs))


def process(loader, state_dict, net):
    torch.cuda.set_device(0)
    net.load_state_dict(state_dict)
    net.eval()
    net.cuda()
    result = {}
    for i, (feature, feature_mask, num_feat, pos_ind, video_id) in enumerate(loader):
        feature = feature[0].cuda()
        feature_mask = feature_mask[0].cuda()
        pos_ind = pos_ind[0].cuda()
        video_id = video_id[0]
        with torch.no_grad():
            rois, actness, roi_scores = net(
                feature, pos_ind, feature_mask=feature_mask, test_mode=True)
            rois, actness, roi_scores = rois[0].cpu().numpy(
            ), actness[0].cpu().numpy(), roi_scores[0].cpu().numpy()[:, 1]
            # import pdb; pdb.set_trace()
            outputs = [rois, actness, roi_scores, num_feat]
            result[video_id] = outputs
    return result


if __name__ == '__main__':

    ctx = multiprocessing.get_context('spawn')
    net = BinaryClassifier(num_class, args.num_body_segments,
                           args, dropout=args.dropout, test_mode=True)

    checkpoint = torch.load(args.weights)

    # rename subset test
    if args.subset == 'testing':
        args.subset = 'test'
    print("model epoch {} loss: {}".format(
        checkpoint['epoch'], checkpoint['best_loss']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(
        checkpoint['state_dict'].items())}
    # db = ANetDB.get_db("1.3")
    db = THUMOSDB.get_db()
    val_videos = db.get_subset_videos(args.subset)

    # loader = torch.utils.data.DataLoader(
    #     BinaryDataSet(args.feat_root, args.feat_model, test_prop_file, subset_videos=val_videos,
    #                   exclude_empty=True, body_seg=args.num_body_segments,
    #                   input_dim=args.input_dim, test_mode=True, use_flow=args.use_flow,
    #                   test_interval=args.frame_interval, verbose=False, num_local=args.num_local),
    #     batch_size=1, shuffle=False,
    #     num_workers=8, pin_memory=True)
    # out_dict = process(loader, base_dict, net)

    dataset = BinaryDataSet(args.feat_root, args.feat_model, test_prop_file, subset_videos=val_videos,
                            exclude_empty=True, body_seg=args.num_body_segments,
                            input_dim=args.input_dim, test_mode=True, use_flow=args.use_flow, 
                            only_flow=args.only_flow, test_interval=args.frame_interval, 
                            verbose=False, num_local=args.num_local)

    index_queue = ctx.Queue()
    result_queue = ctx.Queue()
    workers = [ctx.Process(target=runner_func, args=(dataset, base_dict, gpu_list[i % len(gpu_list)], index_queue, result_queue))
               for i in range(args.workers)]

    del net

    max_num = args.max_num if args.max_num > 0 else len(dataset)

    for i in range(max_num):
        index_queue.put(i)

    for w in workers:
        w.daemon = True
        w.start()

    proc_start_time = time.time()
    out_dict = {}
    for i in range(max_num):
        rst = result_queue.get()
        out_dict[rst[0]] = rst[1]
        cnt_time = time.time() - proc_start_time
        # print('video {} done, total {}/{}, average {:.04f} sec/video'.format(i, i + 1,
        #                                                                 max_num,
        #                                                                 float(cnt_time) / (i+1)))
    if args.save_scores is not None:
        save_dict = {k: v for k, v in out_dict.items()}
        import pickle

        pickle.dump(save_dict, open(args.save_scores, 'wb'), 2)
