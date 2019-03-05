import numpy as np
import json
import pandas as pd

def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.

    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
        + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


def wrapper_segment_iou(target_segments, candidate_segments):
    """Compute intersection over union btw segments
    Parameters
    ----------
    target_segments : ndarray
        2-dim array in format [m x 2:=[init, end]]
    candidate_segments : ndarray
        2-dim array in format [n x 2:=[init, end]]
    Outputs
    -------
    tiou : ndarray
        2-dim array [n x m] with IOU ratio.
    Note: It assumes that candidate-segments are more scarce that target-segments
    """
    if candidate_segments.ndim != 2 or target_segments.ndim != 2:
        print("target_size {}, candidate_size {}".format(target_segments.shape, candidate_segments.shape))
        raise ValueError('Dimension of arguments is incorrect')

    n, m = candidate_segments.shape[0], target_segments.shape[0]
    tiou = np.empty((n, m))
    for i in range(m):
        tiou[:, i] = segment_iou(target_segments[i, :], candidate_segments)

    return tiou


def area_under_curve(proposals, ground_truth, max_avg_nr_proposals=None,
                     tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    recall, avg_recall, proposals_per_video = average_recall_vs_avg_nr_proposals(
        proposals, ground_truth, max_avg_nr_proposals=max_avg_nr_proposals,
        tiou_thresholds=tiou_thresholds)

    area_under_curve = np.trapz(avg_recall, proposals_per_video)
    # print(avg_recall, proposals_per_video)

    return area_under_curve, avg_recall, proposals_per_video


def average_recall_vs_avg_nr_proposals(proposals, ground_truth,
                                       max_avg_nr_proposals=None,
                                       tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """ Computes the average recall given an average number 
        of proposals per video.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    proposal : df
        Data frame containing the proposal instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        array with tiou thresholds.

    Outputs
    -------
    recall : 2darray
        recall[i,j] is recall at ith tiou threshold at the jth average number of average number of proposals per video.
    average_recall : 1darray
        recall averaged over a list of tiou threshold. This is equivalent to recall.mean(axis=0).
    proposals_per_video : 1darray
        average number of proposals per video.
    """

    # Get list of videos.
    video_lst = ground_truth['video-id'].unique()

    if not max_avg_nr_proposals:
        max_avg_nr_proposals = float(proposals.shape[0])/video_lst.shape[0]

    ratio = max_avg_nr_proposals*float(video_lst.shape[0])/proposals.shape[0]
    # print(ratio)

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')
    proposals_gbvn = proposals.groupby('video-id')

    # For each video, computes tiou scores among the retrieved proposals.
    score_lst = []
    total_nr_proposals = 0
    for videoid in video_lst:
        # Get ground-truth instances associated to this video.
        ground_truth_videoid = ground_truth_gbvn.get_group(videoid)
        this_video_ground_truth = ground_truth_videoid.loc[:, [
            't-start', 't-end']].values

        # Get proposals for this video.
        try:
            proposals_videoid = proposals_gbvn.get_group(videoid)
        except:
            n = this_video_ground_truth.shape[0]
            score_lst.append(np.zeros((n, 1)))
            continue

        this_video_proposals = proposals_videoid.loc[:, [
            't-start', 't-end']].values

        if this_video_proposals.shape[0] == 0:
            n = this_video_ground_truth.shape[0]
            score_lst.append(np.zeros((n, 1)))
            continue

        # Sort proposals by score.
        sort_idx = proposals_videoid['score'].argsort()[::-1]
        this_video_proposals = this_video_proposals[sort_idx, :]

        if this_video_proposals.ndim != 2:
            this_video_proposals = np.expand_dims(this_video_proposals, axis=0)
        if this_video_ground_truth.ndim != 2:
            this_video_ground_truth = np.expand_dims(
                this_video_ground_truth, axis=0)

        nr_proposals = np.minimum(
            int(this_video_proposals.shape[0] * ratio), this_video_proposals.shape[0])
        total_nr_proposals += nr_proposals
        this_video_proposals = this_video_proposals[:nr_proposals, :]

        # Compute tiou scores.
        tiou = wrapper_segment_iou(
            this_video_proposals, this_video_ground_truth)
        score_lst.append(tiou)

    # Given that the length of the videos is really varied, we
    # compute the number of proposals in terms of a ratio of the total
    # proposals retrieved, i.e. average recall at a percentage of proposals
    # retrieved per video.

    # Computes average recall.
    pcn_lst = np.arange(1, 101) / 100.0 * (max_avg_nr_proposals *
                                           float(video_lst.shape[0])/total_nr_proposals)
    matches = np.empty((video_lst.shape[0], pcn_lst.shape[0]))
    positives = np.empty(video_lst.shape[0])
    recall = np.empty((tiou_thresholds.shape[0], pcn_lst.shape[0]))
    # Iterates over each tiou threshold.
    for ridx, tiou in enumerate(tiou_thresholds):

        # Inspect positives retrieved per video at different
        # number of proposals (percentage of the total retrieved).
        for i, score in enumerate(score_lst):
            # Total positives per video.
            positives[i] = score.shape[0]
            # Find proposals that satisfies minimum tiou threshold.
            true_positives_tiou = score >= tiou
            # Get number of proposals as a percentage of total retrieved.
            pcn_proposals = np.minimum(
                (score.shape[1] * pcn_lst).astype(np.int), score.shape[1])

            for j, nr_proposals in enumerate(pcn_proposals):
                # Compute the number of matches for each percentage of the proposals
                matches[i, j] = np.count_nonzero(
                    (true_positives_tiou[:, :nr_proposals]).sum(axis=1))

        # Computes recall given the set of matches per video.
        recall[ridx, :] = matches.sum(axis=0) / positives.sum()

    # Recall is averaged.
    avg_recall = recall.mean(axis=0)

    # Get the average number of proposals per video.
    proposals_per_video = pcn_lst * \
        (float(total_nr_proposals) / video_lst.shape[0])

    return recall, avg_recall, proposals_per_video


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_names = []
    index = 0
    for node1 in data['taxonomy']:
        is_leaf = True
        for node2 in data['taxonomy']:
            if node2['parentId'] == node1['nodeId']:
                is_leaf = False
                break
        if is_leaf:
            class_names.append(node1['nodeName'])

    class_labels_map = {}

    for i, class_name in enumerate(class_names):
        class_labels_map[class_name] = i + 1
    class_labels_map['background'] = 0

    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []
    duration_dict = {}

    for key, value in data['database'].items():
        this_subset = value['subset']
        if subset != 'all':
            if this_subset == subset:
                if subset == 'testing':
                    video_names.append('{}'.format(key))
                    duration_dict['{}'.format(key)] = float(value['duration'])
                else:
                    video_names.append('{}'.format(key))
                    annotations.append(value['annotations'])
                    duration_dict['{}'.format(key)] = float(value['duration'])
        elif subset == 'trainval':
            if this_subset != 'testing':
                video_names.append('{}'.format(key))
                annotations.append(value['annotations'])
                duration_dict['{}'.format(key)] = float(value['duration'])
        else:
            video_names.append('{}'.format(key))
            annotations.append(value['annotations'])
            duration_dict['{}'.format(key)] = float(value['duration'])

    return video_names, annotations, duration_dict


def grd_activity(annotation_path, subset='validation'):
    data = load_annotation_data(annotation_path)
    video_names, annotations, duration_dict = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)

    video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        sample_annots = annotations[i]
        for k, annotation in enumerate(sample_annots):
            begin_t = annotation['segment'][0]
            end_t = annotation['segment'][1]
            t_start_lst.append(begin_t)
            t_end_lst.append(end_t)
            label_lst.append(class_to_idx[annotation['label']])
            video_lst.append(video_names[i])

    ground_truth = pd.DataFrame({'video-id': video_lst,
                                 't-start': t_start_lst,
                                 't-end': t_end_lst})

    return ground_truth, class_to_idx

def get_thumos_names_and_annotations(data, subset):
    video_names = []
    annotations = []
    class_to_idx = data['cls2idx']

    for key, value in data['database'].items():
        this_subset = value['subset']
        if subset != 'all':
            if this_subset == subset:
                video_names.append(key)
                annotations.append(value['annotations'])
        else:
            video_names.append(key)
            annotations.append(value['annotations'])

    return video_names, annotations, class_to_idx

def grd_thumos(annotation_path, subset='testing'):
    data = load_annotation_data(annotation_path)
    video_names, annotations, class_to_idx = get_thumos_names_and_annotations(data, subset)

    video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        sample_annots = annotations[i]
        for k, annotation in enumerate(sample_annots):
            if annotation['label'] >= 0:
                begin_t = annotation['segment'][0]
                end_t = annotation['segment'][1]
                t_start_lst.append(begin_t)
                t_end_lst.append(end_t)
                label_lst.append(annotation['label'])
                video_lst.append(video_names[i])

    ground_truth = pd.DataFrame({'video-id': video_lst,
                                 't-start': t_start_lst,
                                 't-end': t_end_lst,
                                 'label': label_lst})

    return ground_truth, class_to_idx
