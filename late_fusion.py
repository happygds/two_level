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
parser.add_argument("--dataset", type=str, default='activitynet',
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


def iou(target_segments, test_segments):
    """Compute intersection over union btw segments

    Parameters
    ----------
    target_segments : ndarray
        2-dim array in format [m x 2:=[init, end]]
    test_segments : ndarray
        2-dim array in format [n x 2:=[init, end]]

    Outputs
    -------
    iou : ndarray
        2-dim array [m x n] with IOU ratio.

    Note: It assumes that target-segments are more scarce that test-segments

    """
    if target_segments.ndim != 2 or test_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')

    nonzero_target = np.nonzero(
        target_segments[:, 1] - target_segments[:, 0])[0]

    m, n = target_segments.shape[0], test_segments.shape[0]
    iou = np.zeros((m, n))
    for _, i in enumerate(nonzero_target):
        tt1 = np.maximum(target_segments[i, 0], test_segments[:, 0])
        tt2 = np.minimum(target_segments[i, 1], test_segments[:, 1])

        # Non-negative overlap score
        intersection = (tt2 - tt1).clip(0)
        union = ((test_segments[:, 1] - test_segments[:, 0]) +
                 (target_segments[i, 1] - target_segments[i, 0]) -
                 intersection)
        # Compute overlap as the ratio of the intersection
        # over union of two segments at the frame level.
        iou[i, :] = intersection / union
    return iou


# video_list = db.get_subset_videos(args.subset)
# video_list = [v for v in video_list if v.instances != []]
# print("video list size: {}".format(len(video_list)))
# video_list = [compute_frame_count(v, args.frame_path, 'frame*.jpg') for v in video_list]
video_list = pickle.load(open('./video_list', 'rb'))

# load scores
print('loading scores...')
score_list = []
for fname in args.score_files:
    score_list.append(pickle.load(open(fname, 'rb')))
print('load {} piles of scores'.format(len(score_list)))
N = len(score_list)
# assert N == 2

# bottom-up generate proposals
print('generating proposals')

# weights_list = list(np.arange(0.1, 1.01, 0.1)) + list(np.arange(1., 5.01, 1.))
weights_list = [0.5]
for merge_weight in weights_list:
    for v in video_list:
        v.merge_weight = merge_weight
    pr_dict = {}
    pr_score_dict = {}
    score_dict = score_list[0]
    def gen_prop(v):
        if (args.dataset == 'activitynet') or (args.dataset == 'thumos14'):
            vid = v.id
        else:
            vid = v.path.split('/')[-1].split('.')[0]
        rois, actness, roi_scores, frm_cnt = score_list[0][vid]
        # merge other pkl files
        for i in range(1, N):
            this_rois, this_actness, this_roi_scores, _ = score_list[i][vid]
            this_ious = iou(rois, this_rois)
            max_ious, argmax_ious = this_ious.max(axis=1), this_ious.argmax(axis=1)
            sel_rois, sel_actness, sel_roi_scores = this_rois[argmax_ious],\
                this_actness[argmax_ious], this_roi_scores[argmax_ious]
            actness = (actness + v.merge_weight * sel_actness) / (1. + v.merge_weight) * (actness > 0.)
            roi_scores = (roi_scores + merge_weight * sel_roi_scores) / (1. + merge_weight) * (actness > 0.)
            rois = (rois + v.merge_weight * sel_rois) / (1. + v.merge_weight) * (actness > 0.).reshape((-1, 1))
        # actness, roi_scores = actness ** (1./N), roi_scores ** (1./N)

        bboxes = [(roi[0] / float(frm_cnt) * v.duration, roi[1] / float(frm_cnt) * v.duration,
                1, act_score * roi_score, roi_score)
                for (roi, act_score, roi_score) in zip(rois, actness, roi_scores)]
        # filter out too short proposals
        bboxes = list(filter(lambda b: b[1] - b[0] > args.minimum_len, bboxes))
        bboxes = list(filter(lambda b: b[4] > 0.*roi_scores.max(), bboxes))

        # ori_rois, ori_actness, ori_roi_scores, ori_frm_cnt = score_list[1][vid]
        # ori_bboxes = [(roi[0] * v.frame_interval / float(v.frame_cnt) * v.duration,
        #                roi[1] * v.frame_interval / float(v.frame_cnt) * v.duration,
        #                1, roi_score*act_score, roi_score)
        #               for (roi, act_score, roi_score) in zip(ori_rois, ori_actness, ori_roi_scores)]
        # # filter out too short proposals
        # ori_bboxes = list(filter(lambda b: b[1] - b[0] <= args.minimum_len, ori_bboxes))
        # bboxes = bboxes + ori_bboxes

        # bboxes = temporal_nms(bboxes, 1. - 1.e-16)
        # bboxes = Soft_NMS(bboxes, length=v.duration)

        if len(bboxes) == 0:
            bboxes = [(0, v.duration, 1, 1)]

        pr_box = [(x[0], x[1]) for x in bboxes]
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
    handle = [pool.apply_async(gen_prop, args=(
        x, ), callback=call_back) for x in video_list]
    pool.close()
    pool.join()

    # evaluate proposal info
    proposal_list = [pr_dict[v.id] for v in video_list if v.id in pr_dict]
    gt_spans_full = [[(x.num_label, x.time_span) for x in v.instances]
                    for v in video_list if v.id in pr_dict]
    gt_spans = [[item[1] for item in x] for x in gt_spans_full]
    new_score_list = [score_dict[v.id] for v in video_list if v.id in pr_dict]
    duration_list = [v.duration for v in video_list if v.id in pr_dict]
    proposal_score_list = [pr_score_dict[v.id]
                        for v in video_list if v.id in pr_dict]
    # print('{} groundtruth boxes from'.format(sum(map(len, gt_spans))))
    # # import pdb ; pdb.set_trace()

    # print('average # of proposals: {}'.format(
    #     np.mean(list(map(len, proposal_list)))))
    IOU_thresh = np.arange(0.5, 1.0, 0.05)
    p_list = []
    for th in IOU_thresh:
        pv, pi = get_temporal_proposal_recall(proposal_list, gt_spans, th)
        # print('IOU threshold {}. per video recall: {:02f}, per instance recall: {:02f}'.format(
        #     th, pv * 100, pi * 100))
        p_list.append((pv, pi))
    # print('Average Recall: {:.04f} {:.04f}'.format(*(np.mean(p_list, axis=0)*100)))

    if args.write_proposals:

        name_pattern = 'frame*.jpg'
        frame_path = args.frame_path

        named_proposal_list = [name_proposal(
            x, y) for x, y in zip(gt_spans_full, proposal_list)]
        # allow_empty = args.dataset == 'activitynet' and args.subset == 'testing'
        dumped_list = [dump_window_list(v, prs, frame_path, name_pattern, score=score, allow_empty=True) for v, prs, score in
                    zip(filter(lambda x: x.id in pr_dict, video_list), named_proposal_list, new_score_list)]

        with open(args.write_proposals, 'w') as of:
            for i, e in enumerate(dumped_list):
                of.write('# {}\n'.format(i + 1))
                of.write(e)

        # print('list {} written. got {} videos'.format(
        #     args.write_proposals, len(dumped_list)))


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
    dir_path = os.path.split(args.score_files[0])[0]
    prediction.to_csv('val.csv')

    # prediction.to_csv(os.path.join(opt.result_path, '{}.csv'.format('val')))
    ground_truth, cls_to_idx = grd_activity(
        'data/activity_net.v1-3.min_save.json', subset='validation')
    del cls_to_idx['background']
    auc, ar_at_prop, nr_proposals_lst = area_under_curve(prediction, ground_truth, max_avg_nr_proposals=100,
                                                        tiou_thresholds=np.linspace(0.5, 0.95, 10))
    nr_proposals_lst = np.around(nr_proposals_lst)

    # for j, nr_proposals in enumerate(nr_proposals_lst[::5]):
    #     print('AR@AN({}) is {}'.format(int(nr_proposals), ar_at_prop[j*5]))
    # print('AR@1 is {:.6f}, AR@10 is {:.6f}, AR@20 is {:.6f}'.format(
    #     ar_at_prop[0], ar_at_prop[9], ar_at_prop[19]))
    print('merge_weight {:.2f}, AR@50 is {:.6f}, AR@100 is {:.6f}, AUC is {:.6f}'.format(
        merge_weight, ar_at_prop[49], ar_at_prop[99], auc))
