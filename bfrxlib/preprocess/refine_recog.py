import glob
import os
import torch
import cv2
import numpy as np
from tqdm import tqdm

from facexlib.detection import init_detection_model
from facexlib.utils.face_restoration_helper import get_largest_face

from bfrxlib.preprocess.models import Descriptor, clip_bbox
from bfrxlib.preprocess.utils import video2images, load_frames
from bfrxlib.preprocess.utils import read_pos_info
from bfrxlib.metrics.basic import cal_similarity

def detect_clip(frame_list, det_net='retinaface_resnet50', only_keep_largest=True):
    det_net = init_detection_model(det_net)
    pos_list = []
    for frame in frame_list:
        with torch.no_grad():
            # BGR inputs/outputs
            bboxes = det_net.detect_faces(frame)
            if len(bboxes) == 0:
                continue
            if only_keep_largest:
                h, w, _ = frame.shape
                _, largest_idx = get_largest_face(bboxes, h, w)
            else:
                largest_idx = 0
        pos_list.append(bboxes[largest_idx])
    return pos_list


def recog_id_switch(frame_list, pos_list, recog_net=None, recog_threshold=1.0, device='cuda'):
    """To refine the tracking results: judge if there is a inter-frame ID-Switch from the tracker based on face recognition.

    Args:
        frame_list (_type_): _description_
        pos_list (_type_): _description_
        det_net (_type_, optional): _description_. Defaults to None.
        recog_net (_type_, optional): _description_. Defaults to None.
        recog_threshold (float, optional): Return the average distance if set to None. Defaults to 1.0.
        device (str, optional): _description_. Defaults to 'cuda'.

    Returns:
        numpy.array: difference/similarty metrix.
    """
    # print(device)
    if recog_net is None:
        recog_net = Descriptor(device=device)
    
    feat_matrix = []
    for frame, pos_info in zip(frame_list, pos_list):
        w, h = frame.shape[:2]
        x1, y1, x2, y2 = clip_bbox(pos_info, w, h)[:4]
        # print('y1, y2', int(y1), int(y2), 'x1, x2', int(x1), int(x2), 'frame shape', frame.shape)
        img = frame[int(y1):int(y2), int(x1):int(x2), :]
        # else:
            # img = wraped_face[0]
        img_t = recog_net._preprocess(img)
        output = recog_net(img_t)
        # print(output)
        feat_matrix.append(output)

    feat_matrix_ori = torch.cat(feat_matrix, dim=0)
    # calculate dist
    first = feat_matrix.pop(0)
    feat_matrix.append(first)
    feat_matrix_shift = torch.cat(feat_matrix, dim=0)
    diff = cal_similarity(feat_matrix_ori, feat_matrix_shift, type='l2')
    diff = diff[:-1]
    
    # print(diff)
    if recog_threshold is not None:
        diff_idx = torch.where(diff > recog_threshold)[0].tolist()
        return diff_idx
    # print('Average ID Similarity score: ', sum(diff) / len(diff))
    return (sum(diff) / len(diff)).cpu().detach().item()


def idswitch2dustbin(args):
    video_list = sorted(glob.glob(os.path.join(args.input_path, '*.mp4')))

    dustbin_dir = makedir((args.input_path, 'dustbin'))
    
    pbar = tqdm(total=len(video_list), unit='clips')
    dustbin_num = 0
    for video_name in video_list:
        det_failure = False
        base_name = os.path.basename(video_name)
        frame_list = video2images(video_name)
        pos_list = detect_clip(frame_list)
        if len(pos_list) != len(frame_list):
            det_failure = True

        id_switches = len(recog_id_switch(frame_list, pos_list))
        if id_switches or det_failure:
            os.rename(video_name, os.path.join(dustbin_dir, base_name))
            dustbin_num += 1
        pbar.set_description(f"dustbin num: {dustbin_num}")
        pbar.update(1)


def calculate_id_similarity_folder(args):
    def get_aligned_pos(frame_list):
        return np.array([[0, 0, frame.shape[1], frame.shape[0]] for frame in frame_list])
    all_frames_dir = os.listdir(args.input_path)
    print('Input: ', args.input_path)

    target = 'final_results'
    # target = 'restored_faces'
    ids_list = []
    pbar = tqdm(total=len(all_frames_dir))
    for frames_dir in all_frames_dir:
        pbar.update(1)
        pbar.set_description(frames_dir)
        frame_list = load_frames(os.path.join(args.input_path, frames_dir, target), name_list=False)
        # frame_list = load_frames(os.path.join(args.input_path, frames_dir), name_list=False)
        # print(frame_list)
        
        if args.aligned:
            pos_info_list = get_aligned_pos(frame_list)
        else:
            pos_info_list = read_pos_info(frames_dir, args.pos_dir) if args.pos_dir is not None else detect_clip(frame_list)
        # print(pos_info_list)
        ids = recog_id_switch(frame_list, pos_info_list, recog_threshold=None)
        ids_list.append(ids)
        # print(ids_list)
        # exit()
    print('Average ID similarity: ', sum(ids_list)/len(ids_list))

if __name__ == '__main__':
    from tqdm import tqdm
    import argparse
    from pathlib import Path

    from bfrxlib.preprocess.utils import makedir

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, required=True)
    parser.add_argument('-s', '--save_label', action='store_true', help='Save pos info of the detected faces.')
    parser.add_argument('-aligned', action='store_true')
    parser.add_argument('-p', '--pos_dir', type=str, default=None)
    args = parser.parse_args()

    # proj_dir = Path(__file__).resolve().parents[2]
    # args.pos_dir = os.path.join(proj_dir, args.pos_dir) if args.pos_dir is not None else None
    if args.aligned:
        print('Calculate ID similarity in aligned faces.')
    elif args.pos_dir is None:
        print('Faces not aligned. Using detector to detect the face bbox.')
    calculate_id_similarity_folder(args)
    