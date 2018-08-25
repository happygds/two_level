import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from ops.sequence_funcs import label_frame_by_threshold, build_box_by_search, temporal_nms
from ops.eval_utils import wrapper_segment_iou


def proposal_layer(score_outputs, gts=None, test_mode=False, ss_prob=0., 
                   rpn_post_nms_top=32, feat_stride=16):
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
    device_id = score_outputs[0].device
    score_outputs = [score_output.data.cpu().numpy() for score_output in score_outputs]
    batch_size = score_outputs[0].shape[0]
    topk_cls = [0]
    tol_lst = [0.05, .1, .2, .3, .4, .5, .6, 0.8, 1.0]
    bw = 3
    thresh=[0.01, 0.05, 0.1, .15, 0.25, .4, .5, .6, .7, .8, .9, .95,]
    rpn_rois = np.zeros((batch_size, rpn_post_nms_top, 3))
    labels = np.zeros((batch_size, rpn_post_nms_top, 2))
    if test_mode:
        actness = np.zeros((batch_size, rpn_post_nms_top))

    for k in range(batch_size):
        # the k-th sample
        bboxes = []
        for s in range(len(score_outputs)):
            scores = score_outputs[s][k]
            # # use TAG
            # topk_labels = label_frame_by_threshold(scores, topk_cls, bw=bw, thresh=thresh, multicrop=False)
            # props = build_box_by_search(topk_labels, np.array(tol_lst))
            # props = [(x[0], x[1], x[3] / float(x[1] - x[0])) for x in props]
            # use change point
            scores = scores[:, 1]
            tmp = gaussian_filter(scores[1:,] - scores[:-1,], bw)
            std_value = tmp.std()
            starts = np.nonzero(tmp > std_value)[0] + 1
            ends = np.nonzero(tmp < -std_value)[0] + 1
            props = [(x, y, 1, scores[x:y].mean()) for x in starts for y in ends if x < y] + [(0, len(scores), 1, scores.mean())]
            bboxes.extend(props)
        # import pdb; pdb.set_trace()
        bboxes = temporal_nms(bboxes, 0.9)[:rpn_post_nms_top]
        rois = [(x[0], x[1]) for x in bboxes]
        rpn_rois[k, :, 0] = k
        rpn_rois[k, :len(bboxes), 1:] = np.asarray(rois)
        if not test_mode:
            # compute iou with ground-truths
            # import pdb; pdb.set_trace()
            gt_k = gts[k]
            gt_k = [x.cpu().numpy() for x in gt_k]
            gt_k = list(filter(lambda b: b[1] + b[0] > 0, gt_k))
            if len(gt_k) == 0:
                gt_k = [(0, 1)]
            gt_k, rois = np.asarray(gt_k), np.asarray(rois)
            rois_iou = wrapper_segment_iou(gt_k, rois)
            m, n = rois_iou.shape
            for i in range(n):
                rois_iou_i = rois_iou[:, i]
                rois_iou[:, i] = (rois_iou_i == rois_iou_i.max()) * rois_iou_i
            rois_iou = (rois_iou.max(axis=1) > 0.7).reshape((-1, 1))
            rois_iou = np.concatenate([1. - rois_iou, rois_iou], axis=1)
            labels[k, :len(bboxes), :] = rois_iou
        else:
            actness[k, :len(bboxes)] = np.asarray([x[2] for x in bboxes])

    # compute mask
    rpn_rois_mask = (np.abs(rpn_rois).mean(axis=2) > 0.).astype('float32')
    rois_relative_pos = np.zeros((batch_size, rpn_post_nms_top, rpn_post_nms_top, 2)).astype('float32')
    # compute geometric attention
    rois_cent, rois_dura = rpn_rois.mean(axis=2), rpn_rois[:, :, 1] - rpn_rois[:, :, 0]
    rois_relative_pos[:, :, :, 0] = np.abs(rois_cent[:, np.newaxis, :] - rois_cent[:, :, np.newaxis]) / rois_dura[:, np.newaxis, :].clip(1e-14)
    rois_relative_pos[:, :, :, 1] = rois_dura[:, :, np.newaxis] / rois_dura[:, np.newaxis, :].clip(1e-14)
    rois_relative_pos = np.log(rois_relative_pos.clip(1e-3)) * rpn_rois_mask[:, :, np.newaxis, np.newaxis] * rpn_rois_mask[:, np.newaxis, :, np.newaxis]

    import pdb; pdb.set_trace()
    # convert numpy to pytorch
    rpn_rois = torch.from_numpy(rpn_rois.astype('float32')).cuda().requires_grad_(False).to(device_id)
    rpn_rois_mask = torch.from_numpy(rpn_rois_mask.astype('float32')).cuda().requires_grad_(False).to(device_id)
    rois_relative_pos = torch.from_numpy(rois_relative_pos.astype('float32')).cuda().requires_grad_(False).to(device_id)

    if not test_mode:
        labels = torch.from_numpy(labels.astype('float32')).cuda().requires_grad_(False).to(device_id)
        return rpn_rois, rpn_rois_mask, rois_relative_pos, labels
    else:
        actness = torch.from_numpy(actness.astype('float32')).cuda().requires_grad_(False).to(device_id)
        return rpn_rois, rpn_rois_mask, rois_relative_pos, actness
