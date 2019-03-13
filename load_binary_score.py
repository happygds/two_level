import torch.utils.data as data

import os
import glob
import pandas as pd
import math
import random
from numpy.random import randint
from ops.io import load_proposal_file
from transforms import *
from ops.utils import temporal_iou
from ops.eval_utils import intersection
from scipy import interpolate


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
    def __init__(self, video_record, frame_path, rgb_csv_path, flow_csv_path, frame_counts=None,
                 use_flow=True, feat_stride=5, sample_duration=100, only_flow=False):
        self._data = video_record
        self.id = self._data.id
        files = glob.glob(os.path.join(frame_path, self.id, 'frame*.jpg'))
        frame_cnt = len(files)
        # frame_cnt = frame_counts[self.id]
        self.frame_cnt = frame_cnt
        vid_name = self.id

        f = pd.read_csv(os.path.join(rgb_csv_path, vid_name+'.csv'))
        rgb_feat = f.values
        if use_flow:
            f = pd.read_csv(os.path.join(flow_csv_path, vid_name+'.csv'))
            flow_feat = f.values
            if only_flow:
                rgb_feat = flow_feat
            else:
                min_len = min(rgb_feat.shape[0], flow_feat.shape[0])
                # both features are 8-frame strided
                assert abs(rgb_feat.shape[0] - flow_feat.shape[0]) <= 1, \
                    "rgb_feat_shp {} not consistent with flow_feat_shp {} for video {}".format(
                        rgb_feat.shape, flow_feat.shape, vid_name)
                rgb_feat = np.concatenate(
                    (rgb_feat[:min_len], flow_feat[:min_len]), axis=1)
        shp = rgb_feat.shape

        self.feat = rgb_feat.astype('float32')
        self.label = np.zeros((rgb_feat.shape[0],), dtype='float32')
        self.starts = np.zeros((rgb_feat.shape[0],), dtype='float32')
        self.ends = np.zeros((rgb_feat.shape[0],), dtype='float32')
        gts = []
        for i, gt in enumerate(self._data.instance):
            begin_ind, end_ind = gt.covering_ratio
            gts.append([frame_cnt * begin_ind / feat_stride,
                        frame_cnt * end_ind / feat_stride])
            # gts.append([sample_duration * begin_ind, sample_duration * end_ind])
            nbegin_ind, nend_ind = int(round(
                frame_cnt * begin_ind / feat_stride)), int(round(frame_cnt * end_ind / feat_stride))
            # nbegin_ind, nend_ind = int(round(sample_duration * begin_ind)), int(round(sample_duration * end_ind))
            self.label[nbegin_ind:nend_ind+1] = 1.
            dura_i = frame_cnt * (end_ind - begin_ind) / feat_stride / 10.
            # dura_i = sample_duration * (end_ind - begin_ind) / 10.
            try:
                if nbegin_ind < nend_ind:
                    start_nbegin, start_nend = int(max(math.floor(frame_cnt * begin_ind / feat_stride - dura_i), 0)), \
                        int(min(math.ceil(frame_cnt * begin_ind /
                                          feat_stride + dura_i), len(self.label)-1))
                    end_nbegin, end_nend = int(max(math.floor(frame_cnt * end_ind / feat_stride - dura_i), 0)), \
                        int(min(math.ceil(frame_cnt * end_ind /
                                          feat_stride + dura_i), len(self.label)-1))
                    # start_nbegin, start_nend = int(max(math.floor(sample_duration * begin_ind - dura_i), 0)), \
                    #             int(min(math.ceil(sample_duration * begin_ind + dura_i), len(self.label)-1))
                    # end_nbegin, end_nend = int(max(math.floor(sample_duration * end_ind - dura_i), 0)), \
                    #             int(min(math.ceil(sample_duration * end_ind + dura_i), len(self.label)-1))
                    self.starts[start_nbegin:start_nend +
                                1], self.ends[end_nbegin:end_nend+1] = 1., 1.
            except IndexError:
                print(len(self.ends), nbegin_ind, nend_ind)
                import pdb
                pdb.set_trace()
        self.gts = np.asarray(gts)


class BinaryDataSet(data.Dataset):

    def __init__(self, feat_root, feat_model, prop_file=None,
                 subset_videos=None, body_seg=5, video_centric=True,
                 test_mode=False, feat_stride=5, input_dim=400,
                 prop_per_video=12, fg_ratio=6, bg_ratio=6,
                 fg_iou_thresh=0.7, bg_iou_thresh=0.01,
                 bg_coverage_thresh=0.02, sample_duration=128*5,
                 gt_as_fg=True, test_interval=6, verbose=True,
                 exclude_empty=True, epoch_multiplier=1, val_mode=False,
                 use_flow=True, only_flow=False, num_local=8,
                 frame_path='/data1/matheguo/important/data/thumos14/frames'):
        self.verbose = verbose
        self.num_local = num_local

        self.body_seg = body_seg
        self.video_centric = video_centric
        self.exclude_empty = exclude_empty
        self.epoch_multiplier = epoch_multiplier
        self.input_dim = input_dim
        self.feat_stride = feat_stride
        self.sample_duration = sample_duration // feat_stride

        self.test_mode = test_mode
        self.val_mode = val_mode
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
        if feat_model == 'feature_anet_200':
            rgb_csv_path = os.path.join(feat_root, 'feature_anet_200/rgb/csv')
            flow_csv_path = os.path.join(
                feat_root, 'feature_anet_200/flow/csv')
            print("using anet_200 feature from {} and {}".format(
                rgb_csv_path, flow_csv_path))
        elif feat_model == 'c3d_feature':
            rgb_csv_path = os.path.join(feat_root, 'feature_c3d_csv')
            flow_csv_path = None
            print("using c3d feature from {}".format(rgb_csv_path))
        else:
            raise NotImplementedError('this feature has been extracted !')

        if prop_file:
            prop_info = load_proposal_file(prop_file)
            frame_counts = {}
            vid_names = []
            for i, vid_info in enumerate(prop_info):
                vid_name = os.path.split(vid_info[0])[1]
                vid_names.append(vid_name)
                frame_counts[vid_name] = int(vid_info[1])
        else:
            frame_counts = None
            raise NotImplementedError('no propfile !')

        if val_mode is True:
            tick_stride = self.sample_duration // 2
        else:
            tick_stride = self.sample_duration // 4

        self.video_list = [BinaryVideoRecord(x, frame_path, rgb_csv_path, flow_csv_path, frame_counts,
                                             use_flow=use_flow, only_flow=only_flow, feat_stride=feat_stride,
                                             sample_duration=self.sample_duration
                                             ) for x in subset_videos if x.id in vid_names]

        count = 0
        if self.test_mode is not True:
            self.video_key_list = {}
            if val_mode:
                self.val_tick_list = {}
            for i, x in enumerate(self.video_list):
                frame_cnt = x.frame_cnt
                frame_ticks = np.arange(
                    0, frame_cnt / feat_stride - self.sample_duration, tick_stride).astype('int32')
                if len(frame_ticks) == 0:
                    frame_ticks = [0]
                for _, frame_tick in enumerate(frame_ticks):
                    self.video_key_list[count] = i
                    if val_mode:
                        self.val_tick_list[count] = frame_tick
                    count += 1
        self.count = max(count, len(self.video_list))

    def __getitem__(self, index):
        real_index = index % self.count
        if self.test_mode:
            return self.get_test_data(self.video_list[real_index])
        elif self.val_mode:
            video_index = self.video_key_list[real_index]
            tick_index = self.val_tick_list[real_index]
            return self.get_training_data(video_index, frame_tick=tick_index)
        else:
            assert real_index in self.video_key_list.keys(
            ), "{} not in video_key_list".format(real_index)
            video_index = self.video_key_list[real_index]
            return self.get_training_data(video_index)

    def _sample_feat(self, feat, label, starts, ends, frame_tick=None):
        feat_num = feat.shape[0]
        if frame_tick is None:
            if feat_num > self.sample_duration:
                begin_index = random.randrange(
                    0, feat_num - self.sample_duration, 4)
            else:
                begin_index = 0
        else:
            begin_index = frame_tick

        out = np.zeros((self.sample_duration, feat.shape[1]), dtype='float32')
        out_label = np.zeros((self.sample_duration,), dtype='float32')
        out_starts = np.zeros((self.sample_duration,), dtype='float32')
        out_ends = np.zeros((self.sample_duration,), dtype='float32')
        min_len = min(feat_num, self.sample_duration)

        out[:min_len] = feat[begin_index:(begin_index+min_len)]
        out_label[:min_len] = label[begin_index:(begin_index+min_len)]
        out_starts[:min_len] = starts[begin_index:(begin_index+min_len)]
        out_ends[:min_len] = ends[begin_index:(begin_index+min_len)]
        assert len(out) == self.sample_duration
        end_ind = begin_index + self.sample_duration

        return out, out_label, out_starts, out_ends, begin_index, end_ind, min_len

    def get_training_data(self, index, frame_tick=None):
        video = self.video_list[index]
        feat = video.feat
        label = video.label
        starts, ends = video.starts, video.ends
        num_feat = feat.shape[0]

        out_feat, out_label, out_starts, out_ends, begin_ind, end_ind, min_len \
            = self._sample_feat(feat, label, starts, ends, frame_tick=frame_tick)
        out_mask = np.zeros_like(out_label).astype('float32')
        out_mask[:min_len] = 1.

        target_segments = video.gts
        test_segments = np.asarray([begin_ind, end_ind]).reshape((-1, 2))
        intersect, ratio_target = intersection(
            target_segments, test_segments, return_ratio_target=True)
        for i, ratio in enumerate(ratio_target[:, 0]):
            this_begin, this_end = target_segments[i]
            this_dura = this_end - this_begin
            start_begin, start_end = this_begin - \
                this_dura / 10., this_begin + this_dura / 10.
            end_begin, end_end = this_end - this_dura / 10., this_end + this_dura / 10.

            this_begin, this_end = max(min(self.sample_duration, int(round(this_begin - begin_ind))), 0), max(
                min(self.sample_duration, int(round(this_end - begin_ind))), 0)
            start_begin, start_end = max(min(self.sample_duration, int(math.floor(start_begin - begin_ind))), 0), max(
                min(self.sample_duration, int(math.floor(start_end - begin_ind))), 0)
            end_begin, end_end = max(min(self.sample_duration, int(math.ceil(end_begin - begin_ind))), 0), max(
                min(self.sample_duration, int(math.ceil(end_end - begin_ind))), 0)
            out_label[this_begin:this_end + 1] = 0.
            out_starts[start_begin:start_end + 1] = 0.
            out_ends[end_begin:end_end + 1] = 0.
            if ratio >= 0.5:
                this_intersect = intersect[i, 0]
                this_begin, this_end = this_intersect[0], this_intersect[1]
                this_dura = this_end - this_begin
                start_begin, start_end = this_begin - \
                    this_dura / 10., this_begin + this_dura / 10.
                end_begin, end_end = this_end - this_dura / 10., this_end + this_dura / 10.

                this_begin, this_end = max(min(self.sample_duration, int(round(this_begin))), 0), max(
                    min(self.sample_duration, int(round(this_end))), 0)
                start_begin, start_end = max(min(self.sample_duration, int(round(start_begin))), 0), max(
                    min(self.sample_duration, int(round(start_end))), 0)
                end_begin, end_end = max(min(self.sample_duration, int(round(end_begin))), 0), max(
                    min(self.sample_duration, int(round(end_end))), 0)
                out_label[this_begin:this_end + 1] = 1.
                out_starts[start_begin:start_end + 1] = 1.
                out_ends[end_begin:end_end + 1] = 1.

        # convert label using haar wavelet decomposition
        gts = np.zeros((256, 2), dtype='float32')
        video_gts = np.zeros((256, 2), dtype='float32')
        assert len(video.gts) <= gts.shape[0], '{} < {}'.format(
            len(video_gts), gts.shape)
        gts[:len(video.gts)] = (video.gts - begin_ind)
        video_gts[:len(video.gts)] = video.gts
        # tmp = ((gts[:, 1] - gts[:, 0]) / (video_gts[:, 1] -
        #                                   video_gts[:, 0]).clip(1e-3) > 0.5).reshape((-1, 1))
        # gts = gts * tmp

        pos_ind = torch.from_numpy(np.arange(begin_ind, end_ind)).long()
        out_feat = torch.from_numpy(out_feat)
        out_label = torch.from_numpy(out_label)
        out_starts, out_ends = torch.from_numpy(
            out_starts), torch.from_numpy(out_ends)
        out_mask = torch.from_numpy(out_mask)

        if frame_tick is None:
            return out_feat, out_mask, out_label, out_starts, out_ends, pos_ind, gts
        else:
            frame_tick = np.array([frame_tick]).reshape((1,)).astype('float32')
            index = np.array([index]).reshape((1,)).astype('float32')
            return out_feat, out_mask, pos_ind, torch.from_numpy(frame_tick), torch.from_numpy(index)

    def get_test_data(self, video, gen_batchsize=1):
        props = []
        video_id = video.id
        feat = video.feat
        frame_cnt = video.frame_cnt

        frame_ticks = np.arange(
            0, feat.shape[0] - self.sample_duration, self.sample_duration // 2).astype('int32')
        if len(frame_ticks) == 0:
            frame_ticks = [0]
        num_sampled_frames = len(frame_ticks)
        # print("the number of samples is {}".format(num_sampled_frames))

        def feat_gen(batchsize):
            feats = []
            seg_inds = []
            cnt = 0
            for idx, seg_ind in enumerate(frame_ticks):
                p = int(seg_ind)
                feats.append(feat[p:min(frame_cnt, p+self.sample_duration)])
                cnt += 1
                seg_inds.append(seg_ind)

                if cnt % batchsize == 0:
                    pos_ind = np.ones(
                        (len(feats), 1)) * np.arange(self.sample_duration).reshape((1, -1))
                    pos_ind = torch.from_numpy(pos_ind).long()

                    out_feat, out_inds = np.stack(feats, axis=0), np.stack(
                        seg_inds, axis=0).reshape((-1,))
                    out_mask = (np.abs(feats).mean(axis=2)
                                > 0.).astype('float32')
                    out_feat = torch.from_numpy(out_feat)
                    out_mask = torch.from_numpy(out_mask)
                    out_inds = torch.from_numpy(out_inds)
                    # print("the count now is {}".format(cnt))
                    yield [out_feat, out_mask, out_inds, pos_ind]
                    feats, seg_inds = [], []

            if len(feats) > 0:
                pos_ind = np.ones((len(feats), 1)) * \
                    np.arange(self.sample_duration).reshape((1, -1))
                pos_ind = torch.from_numpy(pos_ind).long()

                out_feat, out_inds = np.stack(feats, axis=0), np.stack(
                    seg_inds, axis=0).reshape((-1,))
                out_mask = (np.abs(feats).mean(axis=2) > 0.).astype('float32')
                out_feat = torch.from_numpy(out_feat)
                out_mask = torch.from_numpy(out_mask)
                yield [out_feat, out_mask, seg_inds, pos_ind]

        return feat_gen(gen_batchsize), video_id

        # num_feat = feat.shape[0]
        # feats_mask = (np.abs(feats).mean(axis=2) > 0.).astype('float32')
        # out_feat = torch.from_numpy(feats)
        # out_mask = torch.from_numpy(feats_mask)

        # return out_feat, out_mask, num_feat, pos_ind, video_id

    def __len__(self):
        return self.count * self.epoch_multiplier
