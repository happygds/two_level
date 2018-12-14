import numpy as np
import torch
import math
import multiprocessing as mp
from scipy.ndimage import gaussian_filter

from ops.sequence_funcs import label_frame_by_threshold, build_box_by_search, temporal_nms, Soft_NMS
from ops.eval_utils import wrapper_segment_iou


def proposal_layer(score_output, feature_mask, gts=None, test_mode=False, ss_prob=0., 
                   rpn_post_nms_top=100, feat_stride=16, ensemble_stage=None, bboxes=None):
    """
    Parameters
    ----------
    score_outputs: list outputs for multiscale action prediction,
            one element's size (batch_size, timesteps, 2)
    ----------
    Returns
    ----------
    rpn_rois : (batch_size, rpn_post_nms_top, 3) e.g. [0, t1, t2]

    """
    score_output = score_output.data.cpu().numpy()
    feature_mask = feature_mask.data.cpu().numpy()
    batch_size = score_output.shape[0]
    topk_cls = [0]
    tol_lst = [0.05, .1, .2, .3, .4, .5, .6, 0.8, 1.0]
    bw = 3
    thresh=[0.01, 0.05, 0.1, .15, 0.25, .4, .5, .6, .7, .8, .9, .95,]

    if test_mode:
        assert len(feature_mask) == 1
        actness = np.zeros((batch_size, rpn_post_nms_top))
    rpn_rois = np.zeros((batch_size, rpn_post_nms_top, 3))
    start_rois, end_rois = np.zeros_like(rpn_rois), np.zeros_like(rpn_rois)
    labels = np.zeros((batch_size, rpn_post_nms_top, 2))

    for k in range(batch_size):
        # the k-th sample
        num_feat = int(feature_mask[k].sum())
        scores_k = score_output[k][:num_feat]
        # # use TAG
        # scores = scores_k[:, :1]
        # scores = np.concatenate((1-scores, scores), axis=1)
        # topk_labels = label_frame_by_threshold(scores, topk_cls, bw=bw, thresh=thresh, multicrop=False)
        # props = build_box_by_search(topk_labels, np.array(tol_lst))
        # props = [(x[0], x[1], 1, x[3]) for x in props]
        scores = scores_k

        # # use change point
        scores, pstarts, pends = scores[:, 0], scores[:, 1], scores[:, 2]
        if ensemble_stage != '2':
            bboxes = []
            if len(scores) > 1:
                diff_pstarts, diff_pends = pstarts[1:,] - pstarts[:-1,], pends[1:,] - pends[:-1,]
                # gd_scores = gaussian_filter(diff_scores, bw)
                starts = list(np.nonzero((diff_pstarts[:-1] > 0) & (diff_pstarts[1:] < 0))[0] + 1) + list(np.nonzero(pstarts > 0.7 * pstarts.max())[0])
                ends = list(np.nonzero((diff_pends[:-1] > 0) & (diff_pends[1:] < 0))[0] + 1) + list(np.nonzero(pends > 0.7 * pends.max())[0])
                starts, ends = list(set(starts)), list(set(ends))
                props = [(x, y, 1, scores[x:y+1].mean()*(pstarts[x]*pends[y])) for x in starts for y in ends if x < y and scores[x:y+1].mean() > 0.]
                if scores.mean() > 0.:
                    props += [(0, len(scores)-1, 1, scores.mean()*(pstarts[0]*pends[-1]))]
                # import pdb; pdb.set_trace()
            else:
                props = [(0, len(scores)-1, 1, scores.mean()*(pstarts[0]*pends[-1]))]
            # props = [(x[0], x[1], 1, scores[x[0]:x[1]+1].mean()*(pstarts[x[0]]*pends[min(x[1], num_feat-1)])) for x in props]
            bboxes.extend(props)
            bboxes = list(filter(lambda b: b[1] - b[0] > 0, bboxes))
            # to remove duplicate proposals
            bboxes = temporal_nms(bboxes, 1.0 - 1e-14)
            # bboxes = Soft_NMS(bboxes, length=len(scores), max_num=rpn_post_nms_top)[:rpn_post_nms_top]
            if ensemble_stage == '1':
                assert batch_size == 1
                return bboxes
        else:
            if len(bboxes) == 0:
                bboxes = [(0, len(scores)-1, 1, scores.mean()*pstarts[0]*pends[-1])]
            assert bboxes is not None and len(bboxes) > 0
            # to remove duplicate proposals
            bboxes = temporal_nms(bboxes, 1.0 - 1e-14)

        # bboxes = bboxes[:rpn_post_nms_top]
        # bboxes = temporal_nms(bboxes, 0.98)[:rpn_post_nms_top]
        bboxes = Soft_NMS(bboxes, length=len(scores), max_num=rpn_post_nms_top)[:rpn_post_nms_top]
        if len(bboxes) == 0:
            bboxes = [(0, len(scores)-1, 1, scores.mean()*pstarts[0]*pends[-1])]

        rpn_rois[k, :, 0] = k
        rois = [(x[0], x[1]) for x in bboxes]
        rpn_rois[k, :len(bboxes), 1:] = np.asarray(rois)
        start_rois[k, :, 0], end_rois[k, :, 0] = k, k
        rois_begin, rois_end = np.asarray(rois)[:, 0], np.asarray(rois)[:, 1]
        rois_dura = rois_end - rois_begin
        start_rois[k, :len(bboxes), 1], end_rois[k, :len(bboxes), 1] = \
            np.floor(rois_begin - rois_dura / 5.), np.floor(rois_end - rois_dura / 5.)
        start_rois[k, :len(bboxes), 2], end_rois[k, :len(bboxes), 2] = \
            np.ceil(rois_begin + rois_dura / 5.), np.ceil(rois_end + rois_dura / 5.)
        # import pdb; pdb.set_trace()
        if not test_mode:
            # compute iou with ground-truths
            # import pdb; pdb.set_trace()
            gt_k = gts[k]
            gt_k = [x.cpu().numpy() for x in gt_k]
            gt_k = list(filter(lambda b: b[1] + b[0] > 0, gt_k))
            if len(gt_k) == 0:
                gt_k = [(0, 1)]
            gt_k, rois = np.asarray(gt_k), np.asarray(rois)
            rois_iou = wrapper_segment_iou(gt_k, rois).max(axis=1).reshape((-1, 1))
            tmp = np.concatenate([1. - rois_iou, rois_iou], axis=1)
            labels[k, :len(bboxes), :] = tmp
        else:
            actness[k, :len(bboxes)] = np.asarray([x[3] for x in bboxes])
    # compute mask
    rpn_rois_mask = (np.abs(rpn_rois[:, :, 1:]).mean(axis=2) > 0.).astype('float32')
    rois_relative_pos = np.zeros((batch_size, rpn_post_nms_top, rpn_post_nms_top, 2)).astype('float32')
    # compute geometric attention
    rois_cent, rois_dura = rpn_rois[:, :, 1:].mean(axis=2), rpn_rois[:, :, 2] - rpn_rois[:, :, 1]
    rois_start, rois_end = rpn_rois[:, :, 1], rpn_rois[:, :, 2]
    # xx1, yy1 = np.maximum(rois_start[:, np.newaxis, :], rois_start[:, :, np.newaxis]), np.minimum(rois_end[:, np.newaxis, :], rois_end[:, :, np.newaxis])
    # rois_iou = (yy1 - xx1).clip(0.)
    rois_relative_pos[:, :, :, 0] = 20. * (rois_start[:, np.newaxis, :] - rois_start[:, :, np.newaxis]) / rois_dura[:, np.newaxis, :].clip(1e-14)
    rois_relative_pos[:, :, :, 1] = 20. * (rois_end[:, np.newaxis, :] - rois_end[:, :, np.newaxis]) / rois_dura[:, np.newaxis, :].clip(1e-14)
    # rois_relative_pos[:, :, :, 1] = 20. * np.log((np.abs(rois_cent[:, np.newaxis, :] - rois_cent[:, :, np.newaxis]) / rois_dura[:, np.newaxis, :].clip(1e-14)).clip(1e-3))
    # rois_relative_pos[:, :, :, 2] = 20. * np.log((rois_dura[:, :, np.newaxis] / rois_dura[:, np.newaxis, :].clip(1e-14)).clip(1e-3))
    rois_relative_pos = rois_relative_pos * rpn_rois_mask[:, :, np.newaxis, np.newaxis] * rpn_rois_mask[:, np.newaxis, :, np.newaxis]
    # import pdb; pdb.set_trace()

    start_rois = torch.from_numpy(start_rois).cuda().requires_grad_(False).cuda()
    end_rois = torch.from_numpy(end_rois).cuda().requires_grad_(False).cuda()
    rpn_rois = torch.from_numpy(rpn_rois).cuda().requires_grad_(False).float()
    rpn_rois_mask = torch.from_numpy(rpn_rois_mask).cuda().requires_grad_(False).float()
    rois_relative_pos = torch.from_numpy(rois_relative_pos).cuda().requires_grad_(False).float()

    if not test_mode:
        labels = torch.from_numpy(labels).cuda().requires_grad_(False).float()
        return start_rois, end_rois, rpn_rois, rpn_rois_mask, rois_relative_pos, labels
    else:
        actness = torch.from_numpy(actness).cuda().requires_grad_(False).float()
        return start_rois, end_rois, rpn_rois, rpn_rois_mask, rois_relative_pos, actness
