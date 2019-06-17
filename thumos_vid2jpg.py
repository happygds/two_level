from __future__ import print_function, division
import __future__

import os
from subprocess import check_output

import json
import subprocess

def get_video_durations_and_resolution(data_file_path, thumos=False):
    duration_dict = {}
    resolution_dict = {}
    with open(data_file_path, 'r') as data_file:
        data = json.load(data_file)

    for key, value in data['database'].items():
        assert value['duration'] >= 0.
        duration_dict[key] = float(value['duration'])
        resolution_dict[key] = [float(x) for x in value['resolution'].split('x')]
        # seg_times = []
        # for k_annot, annotation in enumerate(value['annotations']):
        #     annot_dura = (annotation['segment'][1] - annotation['segment'][0])
        #     seg_times.append(seg_times)

    return duration_dict, resolution_dict


def resolution(filename):
    """Return resolution of video

    Parameters
    ----------
    filename : str
        Fullpath of video-file

    Outputs
    -------
    width : float
    height : float

    Note: this function makes use of ffprobe and its results depends on it.

    """
    if os.path.isfile(filename):
        cmd = ('ffprobe -v error -show_entries stream=width,height -of default=noprint_wrappers=1' +
               ' ' + filename).split()
        fr_exp = check_output(cmd)
        width_index = fr_exp.find(b'width=')
        height_index = fr_exp.find(b'height=')
        width = eval(compile(fr_exp[(width_index + 6):(width_index + 10)], '<string>', 'eval',
                             __future__.division.compiler_flag))
        height = eval(compile(fr_exp[(height_index + 7):len(fr_exp)], '<string>', 'eval',
                              __future__.division.compiler_flag))
        return width, height
    else:
        raise NotImplementedError('file {} not exist !!! '.format(filename))


if __name__ == "__main__":
    root_path = '../THUMOS14/'
    dst_dir_path = os.path.join(root_path, 'frames')

    subsets = ['video']

    for subset in subsets:
        dir_path = os.path.join(root_path, subset)
        for file_name in os.listdir(dir_path):
            if '.mp4' not in file_name:
                continue
            name, ext = os.path.splitext(file_name)
            vid_name = file_name[:-4]
            dst_directory_path = os.path.join(dst_dir_path, name)

            video_file_path = os.path.join(dir_path, file_name)

            if os.path.exists(dst_directory_path):
                subprocess.call('rm -r {}'.format(dst_directory_path), shell=True)
                print('remove {}'.format(dst_directory_path))
            os.makedirs(dst_directory_path)

            cmd = 'ffmpeg -i {} -vf scale={}:{} {}/frame%06d.jpg'.format(
                video_file_path, 64, 64, dst_directory_path)
            print(cmd)
            subprocess.call(cmd, shell=True)
            print('\n')
