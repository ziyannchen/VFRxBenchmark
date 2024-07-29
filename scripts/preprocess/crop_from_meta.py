import glob
import os
import cv2
from tqdm import tqdm
from bfrxlib.preprocess.utils import video2images


def crop_from_meta(meta_path, data_path, save_cropped_face_dir, is_video_input, resize=-1, save_ext='.png'):
    with open(meta_path, 'r') as f:
        lines = f.read().splitlines()

    meta, frames_info, crop_info = lines[:5], lines[5:-2], lines[-1]
    clip_crop_bbox = crop_info.split(' ')[-4:]
    # print(clip_crop_bbox)
    x0 = int(clip_crop_bbox[0])
    y0 = int(clip_crop_bbox[1])
    x1 = int(clip_crop_bbox[2])
    y1 = int(clip_crop_bbox[3])
    start_frame_idx = int(frames_info[0].split(' ')[0])
    end_frame_idx = int(frames_info[-1].split(' ')[0])

    if is_video_input:
        hq_frame_list = video2images(data_path)[start_frame_idx:end_frame_idx+1]
    else:
        frame_paths = glob.glob(os.path.join(data_path, '*.png'))[start_frame_idx:end_frame_idx+1]
        hq_frame_list = []
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            hq_frame_list.append(hq_frame_list)

    # print(x1-x0, hq_frame_list[0].shape)
    basename = os.path.splitext(os.path.basename(meta_path))[0]
    h, w = hq_frame_list[0].shape[:2]
    assert x1-x0 <= w and y1-y0 <= h, f'mismatch in crop size ({y1-y0}, {x1-x0}) and image size ({h}, {w}) in {basename}, need to compute scale of the cropping box'
    for idx, frame in enumerate(hq_frame_list):
        cropped_face = frame[y0:y1, x0:x1]
        save_cropped_face_path = os.path.join(save_cropped_face_dir, basename) + '_' + str(idx).zfill(4) + save_ext

        if resize != -1:
            cropped_face = cv2.resize(cropped_face, (resize, resize), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(save_cropped_face_path, cropped_face)
        # exit()
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--meta_dir', type=str, required=True)
    parser.add_argument('-d', '--data_dir', type=str, required=True, help='Directory of all clips with frames or directory with video files')
    parser.add_argument('-s', '--save_dir', type=str, required=True)
    parser.add_argument('-resize', '--resize', type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    meta_list = sorted(glob.glob(os.path.join(args.meta_dir, '*')))
    data_list = sorted(os.listdir(args.data_dir))
    ref_list = [d.split('_')[0] for d in data_list] # since the source video file name might be modified when tracking, the source name in the meta file might mismatch the real source name
    data_list = [os.path.join(args.data_dir, d) for d in data_list]
    data_dict = dict(zip(ref_list, data_list))

    is_video_input = True
    if os.path.isdir(data_list[0]):
        # data dir includes splitted frame dirs
        is_video_input = False

    for meta_f in tqdm(meta_list):
        
        clip_name = os.path.splitext(os.path.basename(meta_f))[0]
        data_path = data_dict[clip_name.split('_')[0]]
        save_cropped_face_dir = os.path.join(args.save_dir, clip_name)
        os.makedirs(save_cropped_face_dir, exist_ok=True)
        # print(data_path)
        crop_from_meta(meta_f, data_path, save_cropped_face_dir, is_video_input, resize=args.resize)