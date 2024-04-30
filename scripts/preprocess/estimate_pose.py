import os
import glob
import cv2
import numpy as np
from natsort import natsorted

from bfrxlib.preprocess.utils import split_list_by_n


def standardize(x, x_min=-90, x_max=90):
    '''
    x is original ranged from [min, max], will be scaled into [target_min, 1].
    '''
    target_max = 1
    target_min = 0
    if x_min < 0:
        target_min = -1
    unit = (target_max - target_min ) / (x_max - x_min)
    return x * unit


def cal_pose_var_score(yaw, pitch, roll, yaw_ratio=1.0, pitch_ratio=1.2, roll_ratio=0):
    yaw, pitch, roll = standardize(yaw), standardize(pitch), standardize(roll)

    pose_score = np.array([yaw * yaw_ratio, pitch * pitch_ratio, roll * roll_ratio, .8*(yaw * yaw_ratio + pitch * pitch_ratio + roll * roll_ratio)])
    pose_score = pose_score[np.argmax(abs(pose_score))]
    return 100 * np.clip(pose_score, a_min=-1, a_max=1)

class HeadposeNet:
    def __init__(self) -> None:
        self.net = HeadposeEst()

    def __call__(self, face_img, bbox):
        yaw, pitch, roll = self.net(face_img, bbox=bbox.astype(np.int32))
        score = cal_pose_var_score(yaw[0].item(), pitch[0].item(), roll[0].item())
        return score
    

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


def main(input_dir, save_dir, input_video=False):
    side_face_dir = makedir((save_dir, 'side'))
    normal_face_dir = makedir((save_dir, 'full'))
    # neutral_face_dir = makedir((save_dir, 'neutral'))input_dir

    # score threshold
    side_face_thresh = 40
    full_face_thresh = 30

    model = HeadposeNet()

    if input_video:
        get = get_from_frames_folder
    else:
        get = get_from_img_folder

    pbar = tqdm(total=len(os.listdir(input_dir)))
    all_pose_score = {}
    for path, pos in get(input_dir):
        frame = cv2.imread(path)
        pos = pos if input_video else pos(frame.shape)
        basename = os.path.basename(path)
        headpose_score = model(frame, pos)
        # visualize_headpose(frame, yaw, pitch, roll)
        # cv2.putText(frame, f"{score:.2f}", (10, frame.shape[0] - 40), fontFace=1, fontScale=0.5, color=(255, 255, 255), thickness=1)

        pbar.update(1)
        pbar.set_description(os.path.basename(path)+f': {headpose_score}')

        if abs(headpose_score) < full_face_thresh:
            pass
            os.system('ln -s '+path+' '+os.path.join(normal_face_dir, basename))
            # cv2.imwrite(os.path.join(normal_face_dir, basename), frame)
        elif abs(headpose_score) > side_face_thresh:
            os.system('ln -s '+path+' '+os.path.join(side_face_dir, basename))
            # cv2.imwrite(os.path.join(side_face_dir, basename), frame)
        # else:
            # os.system('ln -s '+path+' '+os.path.join(neutral_face_dir, basename))
        all_pose_score[basename] = headpose_score

    return all_pose_score


if __name__ == '__main__':
    from tqdm import tqdm
    from bfrxlib.preprocess.models import HeadposeEst
    from bfrxlib.preprocess.utils import load_frames, visualize_headpose, makedir

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str)
    parser.add_argument('-o', '--output_dir', type=str)
    parser.add_argument('-v', '--input_video', action='store_true')
    parser.add_argument('-sf', '--save_file', type=str, default=None)
    args = parser.parse_args()

    save_dir = makedir((args.output_dir, 'pose_est'), rebuild=True)
    
    all_pose_score = main(args.input_dir, save_dir, args.input_video)
    
    if args.save_file is not None:
        import json
        res = json.dumps(all_pose_score)
        with open(args.save_file, "w") as f:
            f.write(res)
        
    print('All results saved to '+args.output_dir)
        