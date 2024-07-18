import cv2
import glob
import os
import numpy as np
from io import StringIO
from typing import Tuple
import ffmpeg

from facexlib.detection import init_detection_model

from bfrxlib.utils import yaml_find
from bfrxlib.preprocess.models import SortTracker, Descriptor
from bfrxlib.preprocess.models.utils import add_bbox_margin, clip_bbox
from bfrxlib.preprocess.utils import makedir, obj, append_list_to_dict, iou, enlarge_bbox_square
from bfrxlib.preprocess.utils.video import load_frames, video2images
from bfrxlib.preprocess.refine_recog import recog_id_switch


def clip_refine_and_split(cfg: object, trace_meta_dict: dict, trace_pos_dict: dict):
    # process clips that are too long
    all_trace_ids = trace_meta_dict.keys()
    
    for trace_id in list(all_trace_ids).copy():
        # The trace id reflects the same person in just one single tracking process.
        # Once a trace failed, a afore-tracked face may starts in a new face id.
        meta_list = trace_meta_dict[trace_id]
        face_pos_list = trace_pos_dict[trace_id]
        assert len(meta_list) == len(face_pos_list), 'Error! Length of the meta info is not equal to length of the position info.'

        # Drop possible transition frames in the tail
        if hasattr(cfg.clip, 'drop_trans_frames'):
            meta_list = meta_list[:-cfg.clip.drop_trans_frames]
            face_pos_list = face_pos_list[:-cfg.clip.drop_trans_frames]
        clip_len = len(trace_meta_dict[trace_id])

        if clip_len > cfg.clip.max_len:
            # To split the clip in half
            cur_id = max(all_trace_ids) + 1
            cut_idx = int(clip_len/2)
            meta_left, meta_right = meta_list[:cut_idx], meta_list[cut_idx:]
            pos_left, pos_right = face_pos_list[:cut_idx], face_pos_list[cut_idx:]
            trace_meta_dict[trace_id], trace_pos_dict[trace_id] = meta_left, pos_left
            trace_meta_dict[cur_id], trace_pos_dict[cur_id] = meta_right, pos_right
        
    return trace_meta_dict, trace_pos_dict


def postprocess_trace_info(track_results, frame_w, frame_h, cfg):
    all_trace_meta = {}      # the meta info about the clips refer to the source video
    all_det_pos = {}       # face position info(enlarged bbox, 5 landmarks) refer to the clips

    # add margin to face bboxes & post-process
    for frame_id, (trackers, face_list) in enumerate(track_results):
        # parse frame bbox
        for d in trackers:
            # to find the bbox that matches the tracked one (to get additional landmarks info)
            face_info = face_list[np.argmax(iou(face_list, d))]
            # if face_info[0] - d[0] > 2:
            #     print('coarse or fine bbox', face_info, d)
            det_bbox, face_conf, landmark = face_info[:4], face_info[4], face_info[5:]
            face_pos = np.r_[det_bbox, landmark]

            trace_bbox = d.astype(np.int32)    # [x0, y0, x1, y1, trace_face_id]
            trace_id = trace_bbox[4]
            # drop frame in extremely low resolution
            bbox_w, bbox_h = trace_bbox[2] - trace_bbox[0], trace_bbox[3] - trace_bbox[1]
            if bbox_w < cfg.clip.min_face_res or bbox_h < cfg.clip.min_face_res:
                continue
            
            # add margin to the trace bboxes to get better clips
            bbox_margin = np.min(cfg.clip.bbox_margin_rate * np.array(bbox_h, bbox_w))
            trace_bbox = add_bbox_margin(trace_bbox, bbox_margin, frame_w, frame_h) 

            meta_str = ' '.join([str(frame_id).zfill(6)]+[str(i) for i in trace_bbox[:4]]+[str(face_conf)])
            all_trace_meta = append_list_to_dict(all_trace_meta, key=trace_id, value=meta_str) # det: coarse face bbox (to crop a clip)
            all_det_pos = append_list_to_dict(all_det_pos, key=trace_id, value=face_pos)    # face_pos: fine face position info

    all_trace_meta, all_det_pos = clip_refine_and_split(cfg, all_trace_meta, all_det_pos)
    return all_trace_meta, all_det_pos


def rectify_cropped_clip(frame_list: list, clip_meta_list: list, det_pos_list: list, 
                            save_clip_size: int, frame_w: int, frame_h: int) -> Tuple[list, list]:
    # Post-processing: Get the crop bbox
    trace_frame_str = '\n'.join(clip_meta_list)
    trace_bbox_matrix = np.loadtxt(StringIO(trace_frame_str), delimiter=' ')[:, 1:]   # frame meta matrix as array
    bbox_min, bbox_max = trace_bbox_matrix.min(axis=0), trace_bbox_matrix.max(axis=0)
    crop_bbox = [bbox_min[0], bbox_min[1], bbox_max[2], bbox_max[3]]            # x0, y0, x1, y1
    crop_bbox = enlarge_bbox_square(crop_bbox, max_x=frame_w, max_y=frame_h)    # standardize to a square shape
    # is_large_motion = False
    # if large_face_motion(bbox_matrix, crop_bbox):
        # is_large_motion = True
        # save_clip_dir = os.path.join(save_clip_dir, 'large_motion')

    clip = []
    for frame_idx, (frame_info, face_pos) in enumerate(zip(clip_meta_list, det_pos_list)):
        # frame_info: frame_id, [crop_bbox]
        # face_pos: [bbox], [5 landmarks]
        frame_id = int(frame_info.split(' ')[0])
        frame = frame_list[frame_id][crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]]
        frame = cv2.resize(frame, (save_clip_size, save_clip_size), interpolation=cv2.INTER_CUBIC)
        # cv2.imwrite(args.output_folder+f'/{save_basename}_{str(frame_idx).zfill(3)}.png', frame)

        # print('before scale', face_pos[:4], 'frame w h', crop_bbox)
        # postprocess the bbox & landmarks (only a square-shaped clip size is suppported)
        scale = save_clip_size / (crop_bbox[3] - crop_bbox[1] + 1)
        face_pos[::2] = (face_pos[::2] - crop_bbox[0]) * scale       # for x
        face_pos[1::2] = (face_pos[1::2] - crop_bbox[1]) * scale     # for y
        face_pos[:4] = clip_bbox(np.round(face_pos[:4]), save_clip_size, save_clip_size)
        # print('scaled', face_pos[:4], 'frame w,h', frame_w, frame_h)
        # bad case to drop: the enlarged crop bbox may be shifted, and a large motion clip will not campatible with it.
        
        if face_pos[2] - face_pos[0] <= 0 or face_pos[3] - face_pos[1] <= 0: 
            return [], []

        # face vis
        # visualize_5landmark(frame, face_pos[4:].astype(np.int32))
        # visualize_bbox_rect(frame, face_pos.astype(np.int32))
        
        # face image quality assessment
        # iqa_score = face_iqa(frame, face_pos[:4].astype(np.int32))
        # cv2.putText(frame, str(iqa_score), (int(face_pos[0]) - 10, int(face_pos[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
        #         0.5, (255, 255, 255), 2)

        # headpose estimation
        # yaw, pitch, roll = headpose(frame, face_pos[:4].astype(np.int32))
        clip.append(frame)
    return clip, crop_bbox


def main(frame_list, img_basename, cfg, args, verbose=True):
    # 1. Init
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    clip_dir = makedir(dir_path=(args.output_folder, 'clips'), rebuild=False)   # where the clips are saved
    frame_dir = makedir(dir_path=(args.output_folder, 'frames'), rebuild=False)
    if os.path.exists(os.path.join(clip_dir, img_basename+'.mp4')):
        print('Warning: the clip has been processed, skip...')
        return
    only_keep_largest = hasattr(cfg.det, 'only_keep_largest') and cfg.det.only_keep_largest
    frame_interpolate = hasattr(cfg.det, 'frame_interpolate') and cfg.det.frame_interpolate

    det_net = init_detection_model(cfg.det.model, device=device, half=False)
    tracker = SortTracker(det_net, cfg.det.conf_thresh, detect_interval=args.track_interval)
    if cfg.to_refine:
        recog_net = Descriptor(device=device, input_size=112)

    frame_h, frame_w = frame_list[0].shape[:2]

    # 3. Tracking and processing face clips in videos
    track_results = tracker.track(frame_list=frame_list, only_keep_largest=only_keep_largest)
    all_trace_meta, all_det_pos = postprocess_trace_info(track_results, frame_w, frame_h, cfg)
    
    # 4. Write clips and clip-info into the save folder
    pbar = tqdm(total=len(all_trace_meta.keys()), unit='clips')
    fidx = 0
    for trace_id, clip_meta_list in all_trace_meta.items():
        det_pos_list = all_det_pos[trace_id]

        # Drop clips that are too short
        if len(clip_meta_list) < cfg.clip.min_len:
            if verbose:
                print('Warning, clip is too short to be saved...')
            continue
        
        save_basename = img_basename
        if verbose:
            print('saving to ', f'{clip_dir}/{save_basename}.mp4')
        clip, crop_bbox = rectify_cropped_clip(frame_list, clip_meta_list, det_pos_list, cfg.clip.save_size, frame_w, frame_h)

        if len(clip) == 0:
            continue
        
        if cfg.to_refine:
            id_switches = len(recog_id_switch(clip, det_pos_list, recog_net, device=device))
            # print(id_switches)
            if id_switches:
                continue

        if not only_keep_largest:
            save_basename += f'_{str(fidx).zfill(2)}'
        pbar.set_description(f'Saving {fidx}th clip: {save_basename}')
        fidx += 1

        if args.save_pos:
            # Write pos-info
            pos_dir = makedir(dir_path=(args.output_folder, 'pos_info'), rebuild=False)
            np.savetxt(os.path.join(pos_dir, save_basename+'.txt'), np.round(det_pos_list, 3))

        # Write cropped clips and frames
        target_frame_dir = makedir(dir_path=(frame_dir, save_basename))
        # crop_writer = cv2.VideoWriter(f'{clip_dir}/{save_basename}.mp4', cv2.VideoWriter_fourcc(*'H264'), 
        #                             cfg.clip.save_fps, (cfg.clip.save_size, cfg.clip.save_size))
        # for idx, frame in enumerate(clip):
        #     crop_writer.write(frame)
        #     if idx % cfg.clip.to_frame_interval == 0:
        #         cv2.imwrite(f'{target_frame_dir}/{save_basename}_{str(idx).zfill(4)}.{cfg.db.save_ext}', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # crop_writer.release()
        # ffmpeg_img2video = (
        #         ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{cfg.clip.save_size}x{cfg.clip.save_size}',
        #                      framerate=cfg.clip.save_fps)
        #                     #  .filter('fps', fps=cfg.clip.save_fps, round='up')
        #                      .output('pipe:', format='h264', pix_fmt='yuv420p', vcodec='libx264')
        #                      .run_async(pipe_stdin=True, pipe_stdout=True)
        #                      )
        for idx, frame in enumerate(clip):
            if cfg.clip.to_frame_interval != -1 and idx % cfg.clip.to_frame_interval == 0:
                cv2.imwrite(f'{target_frame_dir}/{save_basename}_{str(idx).zfill(4)}.{cfg.db.save_ext}', frame, cfg.db.imwrite_arg)
            # ffmpeg_img2video.stdin.write(frame.tobytes())
        # ffmpeg_img2video.stdin.close()
        to_video_cmd = f'ffmpeg -y -framerate {cfg.clip.save_fps} -pattern_type glob -i {target_frame_dir}/\'*.{cfg.db.save_ext}\' -pix_fmt yuv420p -c:v libx264 {clip_dir}/{img_basename}.mp4'
        os.system(to_video_cmd)

        if args.save_meta:
            # Write meta-info
            meta_dir = makedir(dir_path=(args.output_folder, 'meta_info'), rebuild=False)
            with open(os.path.join(meta_dir, save_basename + '.txt'), 'w') as save_meta_f:
                save_meta_f.write('Video Source: '+img_basename+'\n')
                # save_meta_f.write(f'Large Motion: {is_large_motion}\n')
                save_meta_f.write(f'W: {frame_w}\nH: {frame_h}\n\n')
                save_meta_f.write('FRAME_ID X0 Y0 X1 Y1 CONF\n')
                # write frame id & bbox info
                save_meta_f.write('\n'.join(clip_meta_list)+'\n\n')
                # write crop rect bbox
                save_meta_f.write(f'CROP_BBOX {crop_bbox[0]} {crop_bbox[1]} {crop_bbox[2]} {crop_bbox[3]}\n')
        
        pbar.update(1)

    print(f'All results are saved to {args.output_folder}')

def parse_input():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', help='Path to the input path (media folder). Support format [.mp4], [.jpg], [.png]', type=str, required=True)
    parser.add_argument('-o', '--output_folder', help='Path to save visualized frames', type=str, default='results')

    parser.add_argument('--track_interval',
        help=('Frame interval to make a detection in tracking, trade-off '
              'between performance and fluency'),
        type=int, default=1)
    parser.add_argument('-cfg', '--config_file', type=str, help="Config file path.", default='ytw.yaml') # bfrxlib/preprocess/cfg
    parser.add_argument('-p', '--prefix', type=str, default='')
    parser.add_argument('--save_meta', action='store_true', help='To save meta info(the clip info in the original raw frames).')
    # parser.add_argument('-l', '--only_keep_largest', action='store_true')
    parser.add_argument('--save_pos', action='store_true', help='To save position info(face position info in every single frame).')

    args = parser.parse_args()

    args.config_file = os.path.join('bfrxlib/preprocess/cfg/', args.config_file)
    return args


if __name__ == '__main__':
    from tqdm import tqdm
    from natsort import natsorted
    import yaml
    import torch
    import warnings
    warnings.filterwarnings('ignore')

    verbose = False
    args = parse_input()
    cfg = obj(yaml.load(open(args.config_file), Loader=yaml.FullLoader))
    args.output_folder = makedir(args.output_folder, rebuild=False)

    # Only accept a directory of frames or video files
    all_file_path = natsorted(os.listdir(args.input_path))
    for idx, input_path in enumerate(all_file_path):
        input_path = os.path.join(args.input_path, input_path)
        print(f'{idx}/{len(all_file_path)}: Processing ', input_path)
        # 2. Read all frames
        if all_file_path[0].endswith(('mp4', 'mov', 'avi')):
            # input path is a dir of videos
            frame_list = video2images(input_path)
            img_basename = os.path.basename(input_path)[:-4].split('_')[0]
            if hasattr(cfg.db, 'id'):
                img_basename = f'{cfg.db.id}_' + img_basename
            main(frame_list, img_basename, cfg, args, verbose)
        elif os.path.isdir(all_file_path[0]):
            # input path is a dir of diffrent frames dirs
            frame_list = load_frames(input_path)
            img_basename = os.path.basename(input_path)
            main(frame_list, img_basename, cfg, args, verbose)
        else:
            # input path is a dir of all images from one clip
            frame_list = [cv2.imread(os.path.join(args.input_path, f), cv2.IMREAD_COLOR) for f in all_file_path]
            clip_basename = os.path.basename(args.input_path)
            main(frame_list, clip_basename, cfg, args, verbose)
            break

    # add verification
    # remove last few frames
