from natsort import natsorted
import os
import glob
import numpy as np

def get_from_frames_folder(frames_folder, frames_interval=10):
    all_clip = os.listdir(os.path.join(frames_folder, 'frames'))
    for clip in all_clip:
        img_dir = os.path.join(frames_folder, 'frames', clip)
        pos_dir = os.path.join(frames_folder, 'pos_info', clip+'.txt')
        pos_info = np.loadtxt(pos_dir)
        # all_frames, frames_path = load_frames(img_dir, name_list=True)
        frames_path = natsorted(glob.glob(os.path.join(img_dir, '*.[jp][pn]g')))
        frames_path = frames_path[::frames_interval]

        for path in frames_path:
            frame_id = int(os.path.splitext(path)[0].split('_')[-1])
            yield path, pos_info[frame_id, ...]

def get_from_img_folder(img_folder):
    all_img_path = natsorted(glob.glob(os.path.join(img_folder, '*.[jp][pn]g')))

    for path in all_img_path:
        yield path, lambda shape: np.array([0, 0, shape[1], shape[0]])

def read_file_recursive(dir_path, all_paths, only_inlcude=None, ignore_name='gt'):
    for entry in os.scandir(dir_path):
        if only_inlcude is not None:
            if entry.name != only_inlcude:
                continue
        elif entry.name == ignore_name:
            # just igname the [ignore_name] folder
            continue
        if entry.is_dir():
            read_file_recursive(entry.path, all_paths)
        else:
            all_paths.append(entry.path)

def path_split(path, split_head):
    assert split_head in path, f'Cannot find {split_head} in {path}'

    target_path = path.split(split_head)[-1]
    if target_path.startswith('/'):
        target_path = target_path[1:]
    return target_path


def uncompress_file(compressed_file, target_dir=None):
    assert os.path.exists(compressed_file)
    if target_dir is None:
        target_dir = os.path.dirname(compressed_file)
        print(f'unzip {compressed_file} {target_dir}/')
        os.system(f'unzip {compressed_file} -d {target_dir}/')