import torch.utils.data as data

import os
import h5py
import os.path
from numpy.random import randint
from ops.io import load_proposal_file
from transforms import *
from ops.utils import temporal_iou


class BinaryInstance:

    def __init__(self, start_frame, end_frame, video_frame_count,
                 fps=1, label=None, overlap_self=None, iou=1.0):
        self.start_frame = start_frame
        self.end_frame = min(end_frame, video_frame_count)
        self._label = label
        self.fps = fps
        self.iou = iou
        self.coverage = (end_frame - start_frame) / video_frame_count

        self.overlap_self = overlap_self

    @property
    def start_time(self):
        return self.start_frame / self.fps

    @property
    def label(self):
        return self._lable if self._label is not None else -1


class BinaryVideoRecord:
    def __init__(self, video_record, frame_path, flow_h5_path, rgb_h5_path,
                 flow_feat_key, rgb_feat_key, use_flow=True, feat_stride=8):
        self._data = video_record
        self.id = self._data.id
        video_name = 'v_{}'.format(self.id)
        files = glob.glob(os.path.join(frame_path, video_name, 'frame*.jpg'))
        frame_cnt = len(files)

        # import pdb
        # pdb.set_trace()
        with h5py.File(rgb_h5_path, 'r') as f:
            rgb_feat = f[vid_name][rgb_feat_key][:][::int(feat_stride // 8)]
        if use_flow:
            with h5py.File(flow_h5_path, 'r') as f:
                flow_feat = f[vid_name][flow_feat_key][:][::int(
                    feat_stride // 8)]
                min_len = min(rgb_feat.shape[0], flow_feat.shape[0])
                # both features are 8-frame strided
                assert abs(rgb_feat.shape[0] - flow_feat.shape[0]) <= 1, \
                    "rgb_feat_shp {} not consistent with flow_feat_shp {} for video {}".format(
                        rgb_feat.shape, flow_feat.shape, vid_name)
                rgb_feat = np.concatenate(
                    (rgb_feat[:min_len], flow_feat[:min_len]), axis=1)
        self.feat = rgb_feat
        
        self.label = np.zeros((rgb_feat.shape[0],), dtype='float32')
        for i, gt in enumerate(self._data.instance):
            begin_ind, end_ind = gt.convering_ratio
            begin_ind, end_ind = int(round(frame_cnt * begin_ind / feat_stride)), int(round(frame_cnt * end_ind / feat_stride))
            self.label[begin_ind:end_ind+1] = 1.


class BinaryDataSet(data.Dataset):

    def __init__(self, feat_root, feat_model,
                 subset_videos=None, body_seg=5, video_centric=True,
                 test_mode=False, feat_stride=16, input_dim=1024,
                 prop_per_video=12, fg_ratio=6, bg_ratio=6,
                 fg_iou_thresh=0.7, bg_iou_thresh=0.01,
                 bg_coverage_thresh=0.02, sample_duration=8196,
                 gt_as_fg=True, test_interval=6, verbose=True,
                 exclude_empty=True, epoch_multiplier=1,
                 use_flow=True, num_local=8, frame_path=None):

        self.subset_videos = subset_videos
        self.verbose = verbose
        self.num_local = num_local

        self.body_seg = body_seg
        self.video_centric = video_centric
        self.exclude_empty = exclude_empty
        self.epoch_multiplier = epoch_multiplier
        self.input_dim = input_dim
        self.feat_stride = feat_stride
        assert feat_stride % 8 == 0
        self.sample_duration = sample_duration // feat_stride

        self.test_mode = test_mode
        self.test_interval = test_interval

        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh

        self.bg_coverage_thresh = bg_coverage_thresh
        self.starting_ratio = 0.5
        self.ending_ratio = 0.5

        self.gt_as_fg = gt_as_fg
        denum = fg_ratio + bg_ratio

        self.fg_per_video = int(prop_per_video * (fg_ratio / denum))
        self.bg_per_video = int(prop_per_video * (bg_ratio / denum))

        # set the directory for the optical-flow features
        if feat_model.endswith('_trained'):
            feat_flow_rpath = os.path.join(feat_root, 'i3d_flow_trained')
        else:
            feat_flow_rpath = os.path.join(feat_root, 'i3d_flow')
        print("using flow feature from {}".format(feat_flow_rpath))

        # obatin the h5 feature directory
        flow_h5_path = os.path.join(feat_flow_rpath, 'i3d_flow_feature.hdf5')
        flow_feat_key = 'i3d_flow_feature'
        feat_rgb_path = os.path.join(feat_root, feat_model)
        if feat_model == 'i3d_rgb' or feat_model == 'i3d_rgb_trained':
            rgb_h5_path = os.path.join(feat_rgb_path, 'i3d_rgb_feature.hdf5')
            rgb_feat_key = 'i3d_rgb_feature'
        elif feat_model == 'inception_resnet_v2':
            rgb_h5_path = os.path.join(
                feat_rgb_path, 'new_inception_resnet.hdf5')
            rgb_feat_key = 'inception_resnet_v2'
        elif feat_model == 'inception_resnet_v2_trained':
            rgb_h5_path = os.path.join(
                feat_rgb_path, 'inception_resnet_v2_trained.hdf5')
            rgb_feat_key = 'inception_resnet_v2'
        else:
            raise NotImplementedError('this feature has been extracted !')
        print("using rgb feature from {}".format(rgb_h5_path))

        self.video_list = [BinaryVideoRecord(x, flow_h5_path, rgb_h5_path, flow_feat_key, rgb_feat_key,
                                             use_flow=use_flow, feat_stride=feat_stride) for x in self.subset_videos]


    def __getitem__(self, index):
        real_index = index % len(self.video_list)
        if self.test_mode:
            return self.get_test_data(self.video_list[real_index])
        else:
            return self.get_training_data(real_index)

    def _sample_frames(self, prop):
        start_frame = prop.start_frame + 1
        end_frame = prop.end_frame
        duration = end_frame - start_frame + 1
        sample_duration = duration / self.body_seg

        if sample_duration < 1:
            return start_frame + randint(prop.end_frame - prop.start_frame, size=self.body_seg)

        frame_indice = []
        split_stage = [int(np.round(i*sample_duration)) +
                       start_frame for i in range(self.body_seg+1)]

        for i in range(self.body_seg):
           #  print(split_stage[i], split_stage[i+1])
            index = np.random.choice(
                range(split_stage[i], split_stage[i+1]), 1)
            frame_indice.extend(index)
        return frame_indice

    def _load_prop_data(self, prop, video_id, begin_ind=0):
        # read frame count
        frame_cnt = self.video_dict[prop[0][0]].num_frames
        # frame_cnt = 1572
        frame_selected = self._sample_frames(prop[0][1])
        # if max(frame_selected) > feat.shape[0] * self.feat_stride:
        #     print(frame_selected, feat.shape[0], video_id)
        frame_selected = [max(min(x, frame_cnt) - 1 - begin_ind, 0) for x in frame_selected]

        return frame_selected, [prop[1]]

    def _video_centric_sampling(self, video, begin_ind=0, end_ind=0):

        fg = video.get_fg(self.fg_iou_thresh, self.gt_as_fg, begin_ind=begin_ind, end_ind=end_ind)
        bg = video.get_bg(self.bg_iou_thresh, begin_ind=begin_ind, end_ind=end_ind)

        def sample_video_proposals(proposal_type, video_id, video_pool, requested_num, dataset_pool):
            if len(video_pool) == 0:
                # if there is noting in the video pool, go fetch from the dataset pool
                return [(dataset_pool[x], 0) for x in np.random.choice(len(dataset_pool), requested_num, replace=False)]
                # return [(dataset_pool[x], proposal_type) for x in np.random.choice(len(dataset_pool), requested_num, replace=False)]
            else:
                replicate = len(video_pool) < requested_num
                idx = np.random.choice(
                    len(video_pool), requested_num, replace=replicate)
                return [((video_id, video_pool[x]), proposal_type) for x in idx]

        out_props = []
        out_props.extend(sample_video_proposals(
            1, video.id, fg, self.fg_per_video, self.fg_pool))  # sample foreground
        out_props.extend(sample_video_proposals(
            0, video.id, bg, self.bg_per_video, self.bg_pool))  # sample background

        return out_props

    def _sample_feat(self, feat, label):
        feat_num = feat.shape[0]
        if feat_num > self.sample_duration:
            begin_index = random.randrange(
                0, feat_num - self.sample_duration + 1, 1)
        else:
            begin_index = 0
        out = np.zeros((self.sample_duration, feat.shape[1]), dtype='float32')
        out_label = np.zeros((self.sample_duration,), dtype='float32')
        min_len = min(feat_num, self.sample_duration)
        out[:min_len] = feat[begin_index:(begin_index+min_len)]
        out_label[:min_len] = label[begin_index:(begin_index+min_len)]
        assert len(out) == self.sample_duration
        end_ind = begin_index + self.sample_duration

        return out, out_label, begin_index, end_ind

    def get_training_data(self, index):
        video = self.video_list[index]
        feat = video.feat
        label = video.label

        out_feats, out_lables, begin_ind, end_ind = self._sample_feat(feat, label)
        pos_ind = torch.from_numpy(np.arange(begin_ind, end_ind)).long()
        out_feats = torch.from_numpy(out_feats)
        out_labels = torch.from_numpy(out_lables)

        # print(out_feats.size(), out_prop_type.size())
        return out_feats, out_label, pos_ind

    def get_test_data(self, video):
        props = []
        video_id = video.id
        feat = video.feat

        frame_ticks = np.arange(feat.shape[0]).astype('int32').reshape((1, -1))
        # num_sampled_frames = len(frame_ticks)
        pos_ind = torch.from_numpy(frame_ticks).long()

        # num_feat = feat.shape[0]
        # if num_feat % self.num_local != 0:
        #     tmp = 8 - num_feat % self.num_local
        #     feat = np.concatenate((feat, np.zeros((tmp, feat.shape[1]), dtype='float32')), dim=1)

        return torch.from_numpy(np.expand_dims(feat, axis=0)), pos_ind

    def __len__(self):
        return len(self.video_list) * self.epoch_multiplier
