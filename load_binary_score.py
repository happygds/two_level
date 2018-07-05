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
    def __init__(self, prop_record, flow_h5_path, rgb_h5_path,
                 flow_feat_key, rgb_feat_key, use_flow=True, feat_stride=8):
        self._data = prop_record
#        print(prop_record)

        frame_count = int(self._data[1])
        self.id = self._data[0]
        vid_name = os.path.split(self._data[0])[1]
        vid_name = 'v_{}'.format(vid_name)
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

        # build instance record
        self.fps = 1
        self.gt = [
            BinaryInstance(int(x[1]), int(x[2]), frame_count, label=int(x[0]), iou=1.0) for x in self._data[2]
            if int(x[2]) > int(x[1])
        ]
        self.gt = list(filter(lambda x: x.start_frame < frame_count, self.gt))

        self.proposals = [
            BinaryInstance(int(x[3]), int(x[4]), frame_count, label=int(x[0]), iou=float(x[1]),
                           overlap_self=float(x[2])) for x in self._data[3] if int(x[4]) > int(x[3])
        ]

        self.proposals = list(
            filter(lambda x: x.start_frame < frame_count, self.proposals))

    @property
    def num_frames(self):
        return int(self._data[1])

    def get_fg(self, fg_thresh, with_gt=True, begin_ind=0, end_ind=0):
        fg = [p for p in self.proposals if p.iou > fg_thresh]
        if with_gt:
            fg.extend(self.gt)
        if begin_ind > 0:
            assert end_ind >  begin_ind
            fg = list(filter(lambda x: x.start_frame < end_ind and x.end_frame > begin_ind, fg))
        return fg

    def get_bg(self, bg_thresh, begin_ind=0, end_ind=0):
        bg = [p for p in self.proposals if p.iou < bg_thresh]
        if begin_ind > 0:
            assert end_ind >  begin_ind
            bg = list(filter(lambda x: x.start_frame < end_ind and x.end_frame > begin_ind, bg))
        return bg


class BinaryDataSet(data.Dataset):

    def __init__(self, feat_root, feat_model,
                 prop_file=None, body_seg=5, video_centric=True,
                 test_mode=False, feat_stride=8, input_dim=1024,
                 prop_per_video=12, fg_ratio=6, bg_ratio=6,
                 fg_iou_thresh=0.7, bg_iou_thresh=0.01,
                 bg_coverage_thresh=0.02, sample_duration=8196,
                 gt_as_fg=True, test_interval=6, verbose=True,
                 exclude_empty=True, epoch_multiplier=1,
                 use_flow=True):

        self.prop_file = prop_file
        self.verbose = verbose

        self.body_seg = body_seg
        self.video_centric = video_centric
        self.exclude_empty = exclude_empty
        self.epoch_multiplier = epoch_multiplier
        self.input_dim = input_dim
        self.feat_stride = feat_stride
        assert feat_stride % 8 == 0
        self.sample_duration = sample_duration // feat_stride

        self.test_mode = test_mode

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

        self._parse_prop_file(flow_h5_path, rgb_h5_path, flow_feat_key, rgb_feat_key,
                              use_flow=use_flow, feat_stride=feat_stride)

    def _parse_prop_file(self, flow_h5_path, rgb_h5_path, flow_feat_key, rgb_feat_key,
                         use_flow=True, feat_stride=8):
        prop_info = load_proposal_file(self.prop_file)

        self.video_list = [BinaryVideoRecord(p, flow_h5_path, rgb_h5_path, flow_feat_key, rgb_feat_key,
                                             use_flow=use_flow, feat_stride=feat_stride) for p in prop_info]

        if self.exclude_empty:
            self.video_list = list(
                filter(lambda x: len(x.gt) > 0, self.video_list))

        self.video_dict = {v.id: v for v in self.video_list}

        # construct two pools:
        # 1. Foreground
        # 2. Background

        self.fg_pool = []
        self.bg_pool = []

        for v in self.video_list:
            self.fg_pool.extend(
                [(v.id, prop) for prop in v.get_fg(self.fg_iou_thresh, self.gt_as_fg)])
            self.bg_pool.extend([(v.id, prop)
                                 for prop in v.get_bg(self.fg_iou_thresh)])

        if self.verbose:
            print("""
            
            BinaryDataSet: Proposal file {prop_file} parse.

            There are {pnum} usable proposals from {vnum} videos.
            {fnum} foreground proposals
            {bnum} background proposals

            Sampling config:
            FG/BG: {fr}/{br}
            
            Epoch size muiltiplier: {em}
            """.format(prop_file=self.prop_file, pnum=len(self.fg_pool) + len(self.bg_pool),
                       fnum=len(self.fg_pool), bnum=len(self.bg_pool),
                       fr=self.fg_per_video, br=self.bg_per_video, vnum=len(
                           self.video_dict),
                       em=self.epoch_multiplier))
        else:
            print("""
                       BinaryDataset: proposal file {prop_file} parsed.
            """.format(prop_file=self.prop_file))
      #  return self.video_list

    def __getitem__(self, index):
        real_index = index % len(self.video_list)
        if self.test_mode:
            return self.get_test_data(self.video_list[real_index], self.test_interval)
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

        return frame_selected, [prop[1]] * len(frame_selected)

    def _video_centric_sampling(self, video, begin_ind=0, end_ind=0):

        fg = video.get_fg(self.fg_iou_thresh, self.gt_as_fg, begin_ind=begin_ind, end_ind=end_ind)
        bg = video.get_bg(self.bg_iou_thresh, begin_ind=begin_ind, end_ind=end_ind)

        def sample_video_proposals(proposal_type, video_id, video_pool, requested_num, dataset_pool):
            if len(video_pool) == 0:
                # if there is noting in the video pool, go fetch from the dataset pool
                return [(dataset_pool[x], proposal_type) for x in np.random.choice(len(dataset_pool), requested_num, replace=False)]
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

    def _sample_feat(self, feat):
        feat_num = feat.shape[0]
        if feat_num > self.sample_duration:
            begin_index = random.randrange(
                0, feat_num - self.sample_duration + 1, 1)
        else:
            begin_index = 0
        out = np.zeros((self.sample_duration, feat.shape[1]), dtype='float32')
        min_len = min(feat_num, self.sample_duration)
        out[:min_len] = feat[begin_index:(begin_index+min_len)]
        assert len(out) == self.sample_duration
        end_ind = begin_index + self.sample_duration

        return out, begin_index, end_ind

    def get_training_data(self, index):

        out_feats, begin_ind, end_ind = self._sample_feat(video.feat)
        pos_ind = torch.from_numpy(np.arange(begin_ind, end_ind+1), requires_grad=True).long()
        video = self.video_list[index]
        props = self._video_centric_sampling(video, begin_ind=begin_ind * self.feat_stride, end_ind=end_ind * self.feat_stride)

        sel_frame_inds = []
        out_prop_type = []

        frames = []
        for idx, p in enumerate(props):
            frame_selected, prop_type = self._load_prop_data(p, video.id, begin_ind=begin_ind)
            sel_frame_inds.extend(frame_selected)
            out_prop_type.extend(prop_type)

        sel_frame_inds = np.around(np.asarray(sel_frame_inds, dtype='float32').reshape(
            (-1, 1)) / self.feat_stride).clip(0., feat.shape[0] - 1)
        sel_frame_inds = np.dot(sel_frame_inds, np.ones(
            (1, self.d_model))).astype('int')
        sel_frame_inds = torch.from_numpy(sel_frame_inds, requires_grad=True).long()

        out_prop_type = torch.from_numpy(np.array(out_prop_type), requires_grad=True).long()
        out_feats = torch.from_numpy(out_feats, requires_grad=True)
        # print(out_feats.size(), out_prop_type.size())
        return out_feats, pos_ind, sel_frame_inds, out_prop_type

    def get_test_data(self, video):
        props = []
        video_id = video.id
        feat = video.feat
        frame_cnt = video.num_frames

        num_sampled_frames = len(frame_ticks)

        frame_ticks = np.arange(feat.shape[0]).astype('int32')
        pos_ind = torch.from_numpy(frame_ticks).long

        # avoid empty proposal list
        for i in frame_ticks:
            props.append(BinaryInstance(i, i+1, 1))

        proposal_tick_list = []

        for proposal in props:
            proposal_ticks = proposal.start_frame, proposal.end_frame
            proposal_tick_list.append(proposal_ticks)

        # load frames
        # Since there are many frames for each video during testing, instead of returning the read frames
        # we return a generator which gives the frames in samll batches, this lower the momeory burden
        # runtime overhead. Usually stting batchsize=4 would fit most cases.

        # def frame_gen(batchsize):
        #     frames= []
        #     cnt = 0
        #     for idx, seg_ind in enumerate(frame_ticks):
        #         p = int(seg_ind)
        #         for x in range(self.new_length):
        #             frames.extend(self._load_image(video_id, min(frame_cnt, p+x)))
        #         cnt += 1

        #         if cnt % batchsize == 0:
        #             frames = self.transform(frames)
        #             yield frames
        #             frames = []

        #     if len(frames):
        #         frames = self.transform(frames)
        #         yield frames

        # return frame_gen(gen_batchsize), len(frame_ticks)

        return torch.from_numpy(feat), pos_ind

    def __len__(self):
        return len(self.video_list) * self.epoch_multiplier
