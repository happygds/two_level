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
from ops.utils import get_actionness_configs, get_reference_model_url

global args
parser = argparse.ArgumentParser(description = 'extract actionnes score')
parser.add_argument('dataset', type=str, choices=['activitynet1.2', 'activitynet1.3', 'thumos14'])
parser.add_argument('subset', type=str, choices=['training','validation','testing'])
parser.add_argument('weights', type=str)
parser.add_argument('save_scores', type=str)
parser.add_argument('--save_raw_scores', type=str, default=None)
parser.add_argument('--frame_interval', type=int, default=16)
parser.add_argument('--test_batchsize', type=int, default=512)
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
parser.add_argument('--feat_model', default='i3d_rgb_trained', type=str,
                    help='the model for extracting pretrained features ('
                    'i3d_rgb | i3d_rgb_trained | inception_resnet_v2 | inception_resnet_v2_trained)')
parser.add_argument('--use_flow', action='store_true',
                    help='whether use i3d_flow feature')
parser.set_defaults(use_flow=True)
parser.add_argument('--dropout', '--do', default=0.8, type=float,
                    metavar='DO', help='dropout ratio (default: 0.8)')

args = parser.parse_args()

dataset_configs = get_actionness_configs(args.dataset)
num_class = dataset_configs['num_class']

if args.dataset == 'thumos14':
    if args.subset == 'validation':
        test_prop_file = 'data/{}_proposal_list.txt'.format(dataset_configs['train_list'])
    elif args.subset == 'testing':
        test_prop_file = 'data/{}_proposal_list.txt'.format(dataset_configs['test_list'])
elif args.dataset == 'activitynet1.2':
    if args.subset == 'training':
        test_prop_file = 'data/{}_proposal_list.txt'.format(dataset_configs['train_list'])
    elif args.subset == 'validation':    
        test_prop_file = 'data/{}_proposal_list.txt'.format(dataset_configs['test_list'])
elif args.dataset == 'activitynet1.3':
    if args.subset == 'training':
        test_prop_file = 'data/{}_proposal_list.txt'.format(dataset_configs['train_list'])
    elif args.subset == 'validation':    
        test_prop_file = 'data/{}_proposal_list.txt'.format(dataset_configs['test_list'])

# set the directory for the rgb features
if args.feat_model == 'i3d_rgb' or args.feat_model == 'i3d_rgb_trained':
    args.input_dim = 1024
elif args.feat_model == 'inception_resnet_v2' or args.feat_model == 'inception_resnet_v2_trained':
    args.input_dim = 1536
if args.use_flow:
    args.input_dim += 1024
print(("=> the input features are extracted from '{}' and the dim is '{}'").format(
    args.feat_model, args.input_dim))

gpu_list = args.gpus if args.gpus is not None else range(8)

def runner_func(dataset, state_dict, gpu_id, index_queue, result_queue):
    torch.cuda.set_device(gpu_id)
    net = BinaryClassifier(num_class, args.num_body_segments, args.input_dim, dropout=args.dropout, test_mode=True)
    # net = torch.nn.DataParallel(net, device_ids=[gpu_id])

    net.load_state_dict(state_dict)
    net.eval()
    net.cuda()
    while True:
        index = index_queue.get()
        frames_gen, frame_cnt = dataset[index]
        input_feat = Variable(frames_gen).cuda()
        with torch.no_grad():
            output, _ = net(input_feat, None)
        print(output[:, 1])
        # output = torch.zeros((frame_cnt, output_dim)).cuda()
        # cnt = 0
        # for frames in frames_gen:
        #     rst, _ = net(input_var, None)
        #     sc = rst.data.view(-1, output_dim)
        #     output[cnt:cnt + sc.size(0), :, :] = sc
        #     cnt += sc.size(0)

        result_queue.put((dataset.video_list[index].id.split('/')[-1], output.cpu().numpy()))
        


if __name__ == '__main__':

    ctx = multiprocessing.get_context('spawn')
    net = BinaryClassifier(num_class, args.num_body_segments, args.input_dim, dropout=args.dropout, test_mode=True)

    checkpoint = torch.load(args.weights)

    print("model epoch {} loss: {}".format(checkpoint['epoch'], checkpoint['best_loss']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
    dataset = BinaryDataSet(args.feat_root, args.feat_model, test_prop_file, 
                            exclude_empty=True, body_seg=args.num_body_segments,
                            input_dim=args.input_dim, test_mode=True, 
                            test_interval=args.frame_interval, verbose=False)

    index_queue = ctx.Queue()
    result_queue = ctx.Queue()
    workers = [ctx.Process(target=runner_func, args=(dataset,base_dict, gpu_list[i % len(gpu_list)], index_queue, result_queue))
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
        print('video {} done, total {}/{}, average {:.04f} sec/video'.format(i, i + 1,
                                                                        max_num,
                                                                        float(cnt_time) / (i+1)))
    if args.save_scores is not None:
        save_dict = {k: v for k,v in out_dict.items()}
        import pickle

        pickle.dump(save_dict, open(args.save_scores, 'wb'), 2)
