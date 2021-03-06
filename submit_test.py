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
from ops.eval_utils import area_under_curve, grd_activity

parser = argparse.ArgumentParser()
parser.add_argument('score_files', type=str, nargs='+')
parser.add_argument("--anet_version", type=str, default='1.3', help='')
parser.add_argument("--dataset", type=str, default='activitynet', choices=['activitynet', 'thumos14'])
parser.add_argument("--cls_scores", type=str, default=None,
                    help='classification scores, if set to None, will use groundtruth labels')
parser.add_argument("--subset", type=str, default='validation', choices=['training', 'validation', 'testing'])
parser.add_argument("--iou_thresh", type=float, nargs='+', default=[0.5, 0.75, 0.95])
parser.add_argument("--score_weights", type=float, nargs='+', default=None, help='')
parser.add_argument("--write_proposals", type=str, default=None, help='')
parser.add_argument("--minimum_len", type=float, default=0, help='minimum length of a proposal, in second')
parser.add_argument("--reg_score_files", type=str, nargs='+', default=None)
parser.add_argument("--frame_path", type=str, default='/mnt/SSD/ActivityNet/anet_v1.2_extracted_340/')
parser.add_argument('--frame_interval', type=int, default=16)

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
        frame_cnt = int(round(video_info.duration * 24))
    video_info.frame_cnt = frame_cnt
    video_info.frame_interval = args.frame_interval
    return video_info
    

video_list = db.get_subset_videos(args.subset)
# video_list = [v for v in video_list if v.instances != []]
print("video list size: {}".format(len(video_list)))
video_list = [compute_frame_count(v, args.frame_path, 'frame*.jpg') for v in video_list]
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
topk = 1
# import pdb
# pdb.set_trace()


def gen_prop(v):
    if (args.dataset == 'activitynet') or (args.dataset == 'thumos14'):
        vid = v.id
    else:
        vid = v.path.split('/')[-1].split('.')[0]
    rois, actness, roi_scores, frm_cnt = score_dict[vid]
    bboxes = [(roi[0], roi[1], 1, roi_score*act_score, roi_score) for (roi, act_score, roi_score) in zip(rois, actness, roi_scores)]
    # filter out too short proposals
    bboxes = list(filter(lambda b: b[1] - b[0] > args.minimum_len, bboxes))
    bboxes = list(filter(lambda b: b[4] > 0.*roi_scores.max(), bboxes))
    # bboxes = temporal_nms(bboxes, 0.9)
    # bboxes = Soft_NMS(bboxes, length=frm_cnt)

    if len(bboxes) == 0:
        bboxes = [(0, float(v.frame_cnt) / v.frame_interval, 1, 1)]

    pr_box = [(x[0] / float(frm_cnt) * v.duration, x[1] / float(frm_cnt) * v.duration) for x in bboxes]
    # pr_box = [(x[0] * v.frame_interval / float(v.frame_cnt) * v.duration, x[1] * v.frame_interval / float(v.frame_cnt) * v.duration) for x in bboxes]

    return v.id, pr_box, [x[3] for x in bboxes]


def call_back(rst):
    pr_dict[rst[0]] = rst[1]
    pr_score_dict[rst[0]] = rst[2]
    import sys
    # print(rst[0], len(pr_dict), len(rst[1]))
    sys.stdout.flush()

pool = mp.Pool(processes=32)
lst = []
handle = [pool.apply_async(gen_prop, args=(x, ), callback=call_back) for x in video_list]
pool.close()
pool.join()

import pandas as pd
video_lst, t_start_lst, t_end_lst, score_lst = [], [], [], []
for k, v in pr_dict.items():
    video_lst.extend([k] * len(v))
    t_start_lst.extend([x[0] for x in v])
    t_end_lst.extend([x[1] for x in v])
    score_lst.extend(pr_score_dict[k])
prediction = pd.DataFrame({'video-id': video_lst,
                            't-start': t_start_lst,
                            't-end': t_end_lst,
                            'score': score_lst})

submit_pred = {}
submit_pred['version'] = "VERSION 1.3"
external_data = {}
external_data['used'] = True
external_data['details'] = "two-stream I3D feature pretrained on kinectics"
submit_pred['external_data'] = external_data

results = {}
vid_names = list(set(video_lst))
for _, vid_name in enumerate(vid_names):
    this_idx = prediction['video-id'] == vid_name
    this_preds = prediction[this_idx][['score', 't-start', 't-end']].values
    this_lst = []
    for _, pred in enumerate(this_preds):
        this_pred = {}
        score, t_start, t_end = pred
        this_pred['score'] = score
        this_pred['segment'] = list([t_start, t_end])
        this_lst.append(this_pred)
    results[vid_name] = this_lst
submit_pred['results'] = results

import json
with open('{}.json'.format('submit_test'), 'w') as outfile:
    json.dump(submit_pred, outfile, indent=4, separators=(',', ': '))