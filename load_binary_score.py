import torch.utils.data as data

import os, glob
import h5py
import math
import random
from numpy.random import randint
from ops.io import load_proposal_file
from transforms import *
from ops.utils import temporal_iou
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
    def __init__(self, video_record, frame_path, flow_h5_path, rgb_h5_path,
                 flow_feat_key, rgb_feat_key, frame_counts=None, use_flow=True, 
                 feat_stride=8, sample_duration=100, only_flow=False):
        self._data = video_record
        self.id = self._data.id
        # files = glob.glob(os.path.join(frame_path, self.id, 'frame*.jpg'))
        # frame_cnt = len(files)
        # frame_cnt = frame_counts[self.id]
        vid_name = 'v_{}'.format(self.id)

        with h5py.File(rgb_h5_path, 'r') as f:
            rgb_feat = f[vid_name][rgb_feat_key][:]
        if use_flow:
            with h5py.File(flow_h5_path, 'r') as f:
                flow_feat = f[vid_name][flow_feat_key][:]
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
        if rgb_feat.shape[0] % 2 != 0:
            rgb_feat = rgb_feat[:-1]
        shp = rgb_feat.shape
        rgb_feat = rgb_feat.reshape((-1, int(feat_stride // 8), shp[1])).mean(axis=1)
        shp = rgb_feat.shape

        # # use linear interpolation to resize the feature into a fixed length
        ori_grids = np.arange(0, shp[0])
        if shp[0] > 1:
            f = interpolate.interp1d(ori_grids, rgb_feat, axis=0)
            x_new=[i*float(shp[0]-1)/(sample_duration-1) for i in range(sample_duration)]
            output = f(x_new)
        else:
            output = np.ones((sample_duration, 1)) * rgb_feat
        rgb_feat = output.astype('float32')
        assert rgb_feat.shape[0] == sample_duration
        
        self.feat = rgb_feat 
        self.label = np.zeros((rgb_feat.shape[0],), dtype='float32')
        self.starts = np.zeros((rgb_feat.shape[0],), dtype='float32')
        self.ends = np.zeros((rgb_feat.shape[0],), dtype='float32')
        gts = []
        for i, gt in enumerate(self._data.instance):
            begin_ind, end_ind = gt.covering_ratio
            # gts.append([frame_cnt * begin_ind / feat_stride, frame_cnt * end_ind / feat_stride])
            gts.append([sample_duration * begin_ind, sample_duration * end_ind])
            # nbegin_ind, nend_ind = int(round(frame_cnt * begin_ind / feat_stride)), int(round(frame_cnt * end_ind / feat_stride))
            nbegin_ind, nend_ind = int(round(sample_duration * begin_ind)), int(round(sample_duration * end_ind))
            self.label[nbegin_ind:nend_ind+1] = 1.
            # dura_i = frame_cnt * (end_ind - begin_ind) / feat_stride / 10.
            dura_i = sample_duration * (end_ind - begin_ind) / 10.
            try:
                if nbegin_ind < nend_ind:
                    # start_nbegin, start_nend = int(max(math.floor(frame_cnt * begin_ind / feat_stride - dura_i), 0)), \
                    #             int(min(math.ceil(frame_cnt * begin_ind / feat_stride + dura_i), len(self.label)-1))
                    # end_nbegin, end_nend = int(max(math.floor(frame_cnt * end_ind / feat_stride - dura_i), 0)), \
                    #             int(min(math.ceil(frame_cnt * end_ind / feat_stride + dura_i), len(self.label)-1))
                    start_nbegin = int(max(round(sample_duration * begin_ind - dura_i), 0))
                    start_nend =  int(min(round(2*sample_duration*begin_ind - start_nbegin), len(self.label)-1))
                    end_nend = int(min(round(sample_duration * end_ind + dura_i), len(self.label)-1))
                    end_nbegin = int(max(round(2*sample_duration*end_ind - end_nend), 0))
                    self.starts[start_nbegin:start_nend+1], self.ends[end_nbegin:end_nend+1] = 1., 1.
            except IndexError:
                print(len(self.ends), nbegin_ind, nend_ind)
                import pdb; pdb.set_trace()
        self.gts = np.asarray(gts)


class BinaryDataSet(data.Dataset):

    def __init__(self, feat_root, feat_model, prop_file=None,
                 subset_videos=None, body_seg=5, video_centric=True,
                 test_mode=False, feat_stride=16, input_dim=1024,
                 prop_per_video=12, fg_ratio=6, bg_ratio=6,
                 fg_iou_thresh=0.7, bg_iou_thresh=0.01,
                 bg_coverage_thresh=0.02, sample_duration=128*16,
                 gt_as_fg=True, test_interval=6, verbose=True,
                 exclude_empty=True, epoch_multiplier=1,
                 use_flow=True, only_flow=False, num_local=8,
                 frame_path='../../data/activitynet/activity_net_frames'):

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

        if prop_file:
            prop_info = load_proposal_file(prop_file)
            frame_counts = {}
            for i, vid_info in enumerate(prop_info):
                vid_name = os.path.split(vid_info[0])[1]
                frame_counts[vid_name] = int(vid_info[1])
        else:
            frame_counts = None
        self.video_list = [BinaryVideoRecord(x, frame_path, flow_h5_path, rgb_h5_path, flow_feat_key, rgb_feat_key,
                                             frame_counts, use_flow=use_flow, only_flow=only_flow, feat_stride=feat_stride, 
                                             sample_duration=self.sample_duration) for x in subset_videos]


    def __getitem__(self, index):
        real_index = index % len(self.video_list)
        if self.test_mode:
            return self.get_test_data(self.video_list[real_index])
        else:
            return self.get_training_data(real_index)

    def _sample_feat(self, feat, label, starts, ends):
        feat_num = feat.shape[0]
        if feat_num > self.sample_duration:
            begin_index = random.randrange(
                0, feat_num - self.sample_duration + 1, 4)
        else:
            begin_index = 0
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
        if out_label[0] == 1.:
            out_starts[0] = 1.
        elif out_label[-1] == 1.:
            out_ends[-1] = 1.

        return out, out_label, out_starts, out_ends, begin_index, end_ind, min_len

    def get_training_data(self, index):
        video = self.video_list[index]
        feat = video.feat
        label = video.label
        starts, ends = video.starts, video.ends
        num_feat = feat.shape[0]

        # sample_duration = num_feat
        # for i, gt in enumerate(video.gts):
        #     begin_ind, end_ind = gt / float(num_feat)
        #     nbegin_ind, nend_ind = int(round(sample_duration * begin_ind)), int(round(sample_duration * end_ind))
        #     label[nbegin_ind:nend_ind+1] = 1.
        #     dura_i = sample_duration * (end_ind - begin_ind) / 10.
        #     try:
        #         if nbegin_ind < nend_ind:
        #             start_nbegin, start_nend = int(max(math.floor(sample_duration * begin_ind - dura_i), 0)), \
        #                         int(min(math.ceil(sample_duration * begin_ind + dura_i), len(label)-1))
        #             end_nbegin, end_nend = int(max(math.floor(sample_duration * end_ind - dura_i), 0)), \
        #                         int(min(math.ceil(sample_duration * end_ind + dura_i), len(label)-1))
        #             starts[start_nbegin:start_nend+1], ends[end_nbegin:end_nend+1] = 1., 1.
        #     except IndexError:
        #         print(len(self.ends), nbegin_ind, nend_ind)
        #         import pdb; pdb.set_trace()

        out_feat, out_label, out_starts, out_ends, begin_ind, end_ind, min_len = self._sample_feat(feat, label, starts, ends)
        out_mask = np.zeros_like(out_label).astype('float32')
        out_mask[:min_len] = 1.

        # convert label using haar wavelet decomposition
        gts = np.zeros((32, 2), dtype='float32')
        gts[:len(video.gts)] = (video.gts - begin_ind).clip(0., min_len)

        pos_ind = torch.from_numpy(np.arange(begin_ind, end_ind)).long()
        out_feat = torch.from_numpy(out_feat)
        out_label = torch.from_numpy(out_label)
        out_starts, out_ends = torch.from_numpy(out_starts), torch.from_numpy(out_ends)
        out_mask = torch.from_numpy(out_mask)

        # print(out_feats.size(), out_prop_type.size())
        return out_feat, out_mask, out_label, out_starts, out_ends, pos_ind, gts

    def get_test_data(self, video):
        props = []
        video_id = video.id
        feat = video.feat

        frame_ticks = np.arange(feat.shape[0]).astype('int32').reshape((1, -1))
        # num_sampled_frames = len(frame_ticks)
        pos_ind = torch.from_numpy(frame_ticks).long()

        # gts = np.zeros((32, 2), dtype='float32')
        # gts[:len(video.gts)] = video.gts

        num_feat = feat.shape[0]
        # if num_feat < 16:
        #     feat = np.concatenate([feat, np.zeros((16-num_feat, feat.shape[1]), dtype='float32')], axis=0)
        feat_mask = (np.abs(feat).mean(axis=1) > 0.).astype('float32')
        out_feat = torch.from_numpy(np.expand_dims(feat, axis=0))
        out_mask = torch.from_numpy(np.expand_dims(feat_mask, axis=0))

        return out_feat, out_mask, num_feat, pos_ind, video_id

    def __len__(self):
        return len(self.video_list) * self.epoch_multiplier
