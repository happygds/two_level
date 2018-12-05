import argparse
import time
import pdb
import torch
import pickle
import numpy as np

from load_binary_score import BinaryDataSet
from binary_model import BinaryClassifier
from transforms import *

from torch import multiprocessing
from torch.utils import model_zoo
from torch.autograd import Variable
from ops.anet_db import ANetDB
from ops.utils import get_actionness_configs, get_reference_model_url

global args
parser = argparse.ArgumentParser(description='extract actionnes score')
parser.add_argument('dataset', type=str, choices=[
                    'activitynet1.2', 'activitynet1.3', 'thumos14'])
parser.add_argument('subset', type=str, choices=[
                    'training', 'validation', 'testing'])
parser.add_argument('weights', type=str)
parser.add_argument('save_scores', type=str)
parser.add_argument('--num_ensemble', type=int, default=10)
parser.add_argument('--save_raw_scores', type=str, default=None)
parser.add_argument('--frame_interval', type=int, default=16)
parser.add_argument('--test_batchsize', type=int, default=32)
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--num_body_segments', type=int, default=5)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--annotation_path', default='/data1/matheguo/important/data/activitynet/activity_net.v1-3.min_save.json',
                    type=str, help='Annotation file path')
parser.add_argument('--feat_root', default='/data1/matheguo/important/data/activitynet',
                    type=str, help='Feature directory path')
parser.add_argument('--result_path', default='/data1/matheguo/important/result/activitynet/self_att',
                    type=str, help='Result directory path')
parser.add_argument('--model', default='TAG', type=str,
                    help='(self_att | TAG')
parser.add_argument('--feat_model', default='i3d_rgb', type=str,
                    help='the model for extracting pretrained features ('
                    'i3d_rgb | i3d_rgb_trained | inception_resnet_v2 | inception_resnet_v2_trained)')
parser.add_argument('--use_flow', action='store_true',
                    help='whether use i3d_flow feature')
parser.set_defaults(use_flow=True)
parser.add_argument('--dropout', '--do', default=0.8, type=float,
                    metavar='DO', help='dropout ratio (default: 0.8)')
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

gpu_list = args.gpus if args.gpus is not None else range(4)

def np_softmax(x, axis=1):
    x_max = np.max(x, axis=axis, keepdims=True)
    x = np.exp(x - x_max)
    x = x / np.sum(x, axis=axis, keepdims=True)
    return x

def runner_func(dataset, state_dict, gpu_id, index_queue, result_queue,
                ensemble_stage, score_output_before):
    torch.cuda.set_device(gpu_id)
    net = BinaryClassifier(num_class, args.num_body_segments,
                           args, dropout=args.dropout, test_mode=True)
    # net = torch.nn.DataParallel(net, device_ids=[gpu_id])

    net.load_state_dict(state_dict)
    net.eval()
    net.cuda()
    while True:
        index = index_queue.get()
        feature, feature_mask, num_feat, pos_ind, video_id = dataset[index]
        feature = feature.cuda()
        feature_mask = feature_mask.cuda()
        pos_ind = pos_ind.cuda()
        video_id = video_id
        with torch.no_grad():
            if ensemble_stage == "1":
                score_output_before = net(feature, pos_ind, feature_mask=feature_mask,
                                          test_mode=True, ensemble_stage=ensemble_stage)
                outputs = score_output_before[0].cpu().numpy()
            elif ensemble_stage == '2':
                score_output = torch.from_numpy(score_output_before[video_id].reshape((1, -1, 3))).cuda()
                rois, actness, roi_scores_before = net(feature, pos_ind, feature_mask=feature_mask,
                                                       test_mode=True, ensemble_stage=ensemble_stage,
                                                       score_output_before=score_output)
                rois, actness, roi_scores_before = rois[0].cpu().numpy(
                ), actness[0].cpu().numpy(), roi_scores_before[0].cpu().numpy()
                # import pdb; pdb.set_trace()
                outputs = [rois, actness, roi_scores_before, num_feat]
            else:
                rois, actness, roi_scores = net(
                    feature, pos_ind, feature_mask=feature_mask, test_mode=True)
                rois, actness, roi_scores = rois[0].cpu().numpy(
                ), actness[0].cpu().numpy(), roi_scores[0].cpu().numpy()[:, 1]
                # import pdb; pdb.set_trace()
                outputs = [rois, actness, roi_scores, num_feat]


        result_queue.put(
            (dataset.video_list[index].id.split('/')[-1], outputs))


if __name__ == '__main__':
    db = ANetDB.get_db("1.3")
    val_videos = db.get_subset_videos(args.subset)
    dataset = BinaryDataSet(args.feat_root, args.feat_model, test_prop_file, subset_videos=val_videos,
                            exclude_empty=True, body_seg=args.num_body_segments,
                            input_dim=args.input_dim, test_mode=True, use_flow=args.use_flow,
                            test_interval=args.frame_interval, verbose=False, num_local=args.num_local)

    # suppose ensemble models from seed1-seedN
    ensemble_stage1 = {}
    for model_id in range(1, args.num_ensemble+1, 1):
        this_path = (args.weights + '.')[:-1]
        this_path = this_path.replace("seed1", "seed"+str(model_id))
        ctx = multiprocessing.get_context('spawn')
        checkpoint = torch.load(this_path)

        print("model epoch {} loss: {}".format(
            checkpoint['epoch'], checkpoint['best_loss']))
        base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(
            checkpoint['state_dict'].items())}

        index_queue = ctx.Queue()
        result_queue = ctx.Queue()
        workers = [ctx.Process(target=runner_func, args=(
            dataset, base_dict, gpu_list[i % len(gpu_list)], index_queue, result_queue, "1", None))
            for i in range(args.workers)]

        max_num = args.max_num if args.max_num > 0 else len(dataset)

        for i in range(max_num):
            index_queue.put(i)

        for w in workers:
            w.daemon = True
            w.start()

        out_stage1 = {}
        for i in range(max_num):
            rst = result_queue.get()
            out_stage1[rst[0]] = rst[1]
        for w in workers:
            w.terminate()
        ensemble_stage1[model_id] = out_stage1

    stage1_outs = {}
    for key in out_stage1.keys():
        for model_id in range(1, args.num_ensemble+1, 1):
            if model_id == 1:
                this_mean = ensemble_stage1[model_id][key] / args.num_ensemble
            else:
                this_mean += ensemble_stage1[model_id][key] / args.num_ensemble
        this_mean = 1. /  (1. + np.exp(-1. * this_mean))
        stage1_outs[key] = this_mean

    # stage 2 : suppose ensemble models from seed1-seedN
    for model_id in range(1, args.num_ensemble+1, 1):
        ensemble_stage2 = {}
        for stage2_id in range(1, args.num_ensemble+1, 1):
            this_path = (args.weights + '.')[:-1]
            this_path = this_path.replace("seed1", "seed"+str(stage2_id))
            ctx = multiprocessing.get_context('spawn')
            checkpoint = torch.load(this_path)

            print("model epoch {} loss: {}".format(
                checkpoint['epoch'], checkpoint['best_loss']))
            base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(
                checkpoint['state_dict'].items())}

            index_queue = ctx.Queue()
            result_queue = ctx.Queue()
            workers = [ctx.Process(target=runner_func, args=(
                dataset, base_dict, gpu_list[i % len(gpu_list)], index_queue, result_queue, "2", ensemble_stage1[model_id]))
                for i in range(args.workers)]

            max_num = args.max_num if args.max_num > 0 else len(dataset)

            for i in range(max_num):
                index_queue.put(i)

            for w in workers:
                w.daemon = True
                w.start()

            out_stage2 = {}
            for i in range(max_num):
                rst = result_queue.get()
                out_stage2[rst[0]] = rst[1]
            ensemble_stage2[stage2_id] = out_stage2
            for w in workers:
                w.terminate()

        stage2_outs = {}
        for key in out_stage2.keys():
            for stage2_id in range(1, args.num_ensemble+1, 1):
                if stage2_id == 1:
                    this_mean = ensemble_stage2[stage2_id][key][2] / args.num_ensemble
                else:
                    this_mean += ensemble_stage2[stage2_id][key][2] / args.num_ensemble
            this_mean = np_softmax(this_mean)[:, 1]
            stage2_outs[key] = ensemble_stage2[stage2_id][key][:2] + [this_mean,] + ensemble_stage2[stage2_id][key][3:]

        if model_id == 1:
            ensemble_outputs = stage2_outs
        else:
            for key, stage2_out in stage2_outs.items():
                last_ensemble_out = ensemble_outputs[key]
                rois = np.concatenate([last_ensemble_out[0], stage2_out[0]], axis=0)
                actness = np.concatenate([last_ensemble_out[1], stage2_out[1]], axis=0)
                roi_scores = np.concatenate([last_ensemble_out[2], stage2_out[2]], axis=0)
                num_feat = last_ensemble_out[3]
                ensemble_outputs[key] = [rois, actness, roi_scores, num_feat]
    
    pickle.dump(ensemble_outputs, open('./ensemble_outputs.pkl', 'wb'), 2)
    import pdb; pdb.set_trace()

    for key, value in ensemble_outputs.items():
        rois, actness, roi_scores, num_feat = value
        scores = stage1_outs[key]
        scores, pstarts, pends = scores[:, 0], scores[:, 1], scores[:, 2]
        actness = [scores[x[0]:x[1]+1].mean()*(pstarts[x[0]]*pends[min(x[1], num_feat-1)]) for x in rois]
        ensemble_outputs[key] = [rois, actness, roi_scores, num_feat]

    if args.save_scores is not None:
        out_dict = ensemble_outputs
        save_dict = {k: v for k, v in out_dict.items()}
        pickle.dump(save_dict, open(args.save_scores, 'wb'), 2)
