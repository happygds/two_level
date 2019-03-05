import argparse
import os
import sys
import glob
import math
import pandas as pd
import numpy as np
import multiprocessing
from sklearn.metrics import confusion_matrix
import time
import pickle
import multiprocessing as mp
from ops.sequence_funcs import *
from ops.anet_db import ANetDB
from ops.thumos_db import THUMOSDB
from ops.detection_metrics import get_temporal_proposal_recall, name_proposal
from ops.sequence_funcs import temporal_nms
from ops.io import dump_window_list
from ops.eval_utils import area_under_curve, grd_thumos

parser = argparse.ArgumentParser()
parser.add_argument('score_files', type=str, nargs='+')
parser.add_argument("--anet_version", type=str, default='1.3', help='')
parser.add_argument("--dataset", type=str, default='thumos14',
                    choices=['activitynet', 'thumos14'])
parser.add_argument("--cls_scores", type=str, default=None,
                    help='classification scores, if set to None, will use groundtruth labels')
parser.add_argument("--subset", type=str, default='validation',
                    choices=['training', 'validation', 'testing'])
parser.add_argument("--iou_thresh", type=float,
                    nargs='+', default=[0.5, 0.75, 0.95])
parser.add_argument("--score_weights", type=float,
                    nargs='+', default=None, help='')
parser.add_argument("--write_proposals", type=str, default=None, help='')
parser.add_argument("--minimum_len", type=float, default=0,
                    help='minimum length of a proposal, in second')
parser.add_argument("--reg_score_files", type=str, nargs='+', default=None)
parser.add_argument("--frame_path", type=str,
                    default='/mnt/SSD/ActivityNet/anet_v1.2_extracted_340/')
parser.add_argument('--frame_interval', type=int, default=5)

args = parser.parse_args()


if args.dataset == 'activitynet':
    db = ANetDB.get_db(args.anet_version)
    db.try_load_file_path('/mnt/SSD/ActivityNet/anet_v1.2_extracted_340/')
elif args.dataset == 'thumos14':
    db = THUMOSDB.get_db()
    db.try_load_file_path('/mnt/SSD/THUMOS14/')

    # rename subset test
    if args.subset == 'testing':
        args.subset = 'test'
else:
    raise ValueError("unknown dataset {}".format(args.dataset))


def compute_frame_count(video_info, frame_path, name_pattern):
    # first count frame numbers
    try:
        video_name = video_info.id
        files = glob.glob(os.path.join(frame_path, video_name, name_pattern))
        frame_cnt = len(files)
    except:
        print("video {} not exist frame images".format(video_info.id))
        import pdb; pdb.set_trace()
        frame_cnt = int(round(video_info.duration * 30))
    video_info.frame_cnt = frame_cnt
    video_info.frame_interval = args.frame_interval
    return video_info

video_list = db.get_subset_videos(args.subset)
video_list = [v for v in video_list if v.instances != []]
print("video list size: {}".format(len(video_list)))
video_list = [compute_frame_count(
    v, args.frame_path, 'frame*.jpg') for v in video_list]
# video_list = pickle.load(open('./video_list', 'rb'))

# load scores
print('loading scores...')
score_list = []
for fname in args.score_files:
    score_list.append(pickle.load(open(fname, 'rb')))
print('load {} piles of scores'.format(len(score_list)))


# load classification scores if specified
if args.cls_scores:
    cls_scores = cPickle.load(open(args.cls_scores, 'rb'))
else:
    cls_scores = None
print('loading clasification score done')

# load regression scores
if args.reg_score_files is not None:
    print('loading regression scores')
    reg_score_list = []
    for fname in args.reg_score_files:
        reg_score_list.append(cPickle.load(open(fname, 'rb')))
    print('load {} piles of regression scores'.format(len(reg_score_list)))
else:
    reg_score_list = None


# merge scores
print('merging scores')
score_dict = {}
# for key in score_list[0].keys():
#     out_score = score_list[0][key].mean(axis=1) * (1.0 if args.score_weights is None else args.score_weights[0])
#     for i in range(1, len(score_list)):
#         add_score = score_list[i][key].mean(axis=1)
#         if add_score.shape[0] < out_score.shape[0]:
#             out_score = out_score[:add_score.shape[0], :]
#         elif add_score.shape[0] > out_score.shape[0]:
#             tick = add_score.shape[0] / float(out_score.shape[0])
#             indices = [int(x * tick) for x in range(out_score.shape[0])]
#             add_score = add_score[indices, :]
#         out_score += add_score * (1.0 if args.score_weights is None else args.score_weights[i])
#     score_dict[key] = out_score
score_dict = score_list[0]
import pdb; pdb.set_trace()
print('done')

# merge regression scores
if reg_score_list is not None:
    print('merging regression scores')
    reg_score_dict = {}
    for key in reg_score_list[0].keys():
        out_score = reg_score_list[0][key].mean(axis=1)
        for i in range(1, len(reg_score_list)):
            add_score = reg_score_list[i][key].mean(axis=1)
            if add_score.shape[0] < out_score.shape[0]:
                out_score = out_score[:add_score.shape[0], :]
            out_score += add_score
        reg_score_dict[key] = out_score / len(reg_score_list)
    print('done')
else:
    reg_score_dict = None

# bottom-up generate proposals
print('generating proposals')
pr_dict = {}
pr_score_dict = {}
pr_fps_dict = {}
topk = 1
# import pdb
# pdb.set_trace()


def gen_prop(v):
    if (args.dataset == 'activitynet') or (args.dataset == 'thumos14'):
        vid = v.id
    else:
        vid = v.path.split('/')[-1].split('.')[0]
    rois, roi_scores = score_dict[vid]
    bboxes = [(roi[0], roi[1], 1, roi_score)
              for (roi, roi_score) in zip(rois, roi_scores)]
    # filter out too short proposals
    bboxes = list(filter(lambda b: b[1] - b[0] > args.minimum_len, bboxes))
    # bboxes = temporal_nms(bboxes, 1e-14)
    # bboxes = Soft_NMS(bboxes, length=frm_cnt)

    if len(bboxes) == 0:
        bboxes = [(0, float(v.frame_cnt) / v.frame_interval, 1, 1)]

    # pr_box = [(x[0] / float(frm_cnt) * v.duration, x[1] / float(frm_cnt) * v.duration) for x in bboxes]
    pr_box = [(x[0] * v.frame_interval / float(v.frame_cnt) * v.duration, x[1]
               * v.frame_interval / float(v.frame_cnt) * v.duration) for x in bboxes]
    pr_fps = float(v.frame_cnt) / v.duration
    print(len(rois), pr_fps)

    return v.id, pr_box, [x[3] for x in bboxes], pr_fps


def call_back(rst):
    pr_dict[rst[0]] = rst[1]
    pr_score_dict[rst[0]] = rst[2]
    pr_fps_dict[rst[0]] = rst[3]
    import sys
    # print(rst[0], len(pr_dict), len(rst[1]))
    sys.stdout.flush()


pool = mp.Pool(processes=32)
lst = []
handle = [pool.apply_async(gen_prop, args=(
    x, ), callback=call_back) for x in video_list]
pool.close()
pool.join()

# evaluate proposal info
proposal_list = [pr_dict[v.id] for v in video_list if v.id in pr_dict]
gt_spans_full = [[(x.num_label, x.time_span) for x in v.instances]
                 for v in video_list if v.id in pr_dict]
gt_spans = [[item[1] for item in x] for x in gt_spans_full]
score_list = [score_dict[v.id] for v in video_list if v.id in pr_dict]
duration_list = [v.duration for v in video_list if v.id in pr_dict]
proposal_score_list = [pr_score_dict[v.id]
                       for v in video_list if v.id in pr_dict]
print('{} groundtruth boxes from'.format(sum(map(len, gt_spans))))
# import pdb ; pdb.set_trace()

print('average # of proposals: {}'.format(
    np.mean(list(map(len, proposal_list)))))
IOU_thresh = np.arange(0.5, 1.0, 0.05)
p_list = []
for th in IOU_thresh:
    pv, pi = get_temporal_proposal_recall(proposal_list, gt_spans, th)
    print('IOU threshold {}. per video recall: {:02f}, per instance recall: {:02f}'.format(
        th, pv * 100, pi * 100))
    p_list.append((pv, pi))
print('Average Recall: {:.04f} {:.04f}'.format(*(np.mean(p_list, axis=0)*100)))

if args.write_proposals:

    name_pattern = 'frame*.jpg'
    frame_path = args.frame_path

    named_proposal_list = [name_proposal(
        x, y) for x, y in zip(gt_spans_full, proposal_list)]
    # allow_empty = args.dataset == 'activitynet' and args.subset == 'testing'
    dumped_list = [dump_window_list(v, prs, frame_path, name_pattern, score=score, allow_empty=True) for v, prs, score in
                   zip(filter(lambda x: x.id in pr_dict, video_list), named_proposal_list, score_list)]

    with open(args.write_proposals, 'w') as of:
        for i, e in enumerate(dumped_list):
            of.write('# {}\n'.format(i + 1))
            of.write(e)

    print('list {} written. got {} videos'.format(
        args.write_proposals, len(dumped_list)))


import pandas as pd
video_lst, t_start_lst, t_end_lst, score_lst = [], [], [], []
f_init_lst, f_end_lst = [], []
for k, v in pr_dict.items():
    video_lst.extend([k] * len(v))
    t_start_lst.extend([x[0] for x in v])
    t_end_lst.extend([x[1] for x in v])
    score_lst.extend(pr_score_dict[k])

    fps = pr_fps_dict[k]
    f_init_lst.extend([x[0] * fps for x in v])
    f_end_lst.extend([x[1] * fps - 1 for x in v])

prediction = pd.DataFrame({'video-id': video_lst,
                           't-start': t_start_lst,
                           't-end': t_end_lst,
                           'score': score_lst})
dir_path = os.path.split(args.score_files[0])[0]
thumos_results = pd.DataFrame({'video-name': video_lst,
                               'f-init': f_init_lst,
                               'f-end': f_end_lst,
                               'score': score_lst})
thumos_results.to_csv('two_level.csv')

# prediction.to_csv(os.path.join(opt.result_path, '{}.csv'.format('val')))
ground_truth, cls_to_idx = grd_thumos(
    'data/thumos_annots.json', subset='testing')
del cls_to_idx['Ambiguous']
auc, ar_at_prop, nr_proposals_lst = area_under_curve(prediction, ground_truth, max_avg_nr_proposals=1000,
                                                     tiou_thresholds=np.linspace(0.5, 0.95, 10))
nr_proposals_lst = np.around(nr_proposals_lst)

for j, nr_proposals in enumerate(nr_proposals_lst[::100]):
    print('AR@AN({}) is {}'.format(int(nr_proposals), ar_at_prop[j*100]))
