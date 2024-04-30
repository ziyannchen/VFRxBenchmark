import torch
from tqdm import tqdm
import os
from os import path as osp

from basicsr.utils import tensor2img
from basicsr.utils import imwrite

from bfrxlib.utils import FaceRestoreHelper

def parse_options(root_path, is_train=False):
    from bfrxlib.utils.options import yaml_load
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    # parser.add_argument('-o', '--output_path', type=str, default='results/tmp', help='Path to save results.')
    # parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    # parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--debug', action='store_true')
    # parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--force_yml', nargs='+', default=None, help='Force to update yml files. Examples: helper:save_ext=jpg')
    args = parser.parse_args()

    # parse yml to dict
    assert osp.exists(args.opt), f'Config file {args.opt} doesn\'t exists!'
    opt = yaml_load(args.opt)

    # force to update yml options
    if args.force_yml is not None:
        for entry in args.force_yml:
            from basicsr.utils.options import _postprocess_yml_value
            # now do not support creating new keys
            keys, value = entry.split('=')
            keys, value = keys.strip(), value.strip()
            value = _postprocess_yml_value(value)
            eval_str = 'opt'
            for key in keys.split(':'):
                eval_str += f'["{key}"]'
            eval_str += '=value'
            # using exec function
            exec(eval_str)

    # TODO: seperate the train opt and inference opt. this is only temporary use.
    opt['is_train'] = is_train
    opt['dist'] = False
    
    # default root dir of results and weights
    # TODO: add README to explain the default path settings
    model_name = opt['name']
    
    if 'logs_root' not in opt['path']:
        opt['path']['logs_root'] = osp.join('logs', model_name)
    opt['path']['logs_root'] = osp.join(root_path, opt['path']['logs_root'])
    os.makedirs(opt['path']['logs_root'], exist_ok=True)

    return opt, args

def inference_image_pipeline(data, model, opt, device):
    results = {'restored_faces': [], 'restored_img': None}
    data_opt = opt['dataset']
    aligned = data_opt['aligned']

    # ------------------ set up FaceRestoreHelper -------------------
    face_helper = FaceRestoreHelper(device=device, face_size=data_opt['face_size'], **opt['helper'])
    # TODO: to support backgrouns upsampler
    bg_upsampler = None

    for lq_img, lq_basename, lq_path, is_gray in zip(data['lq'], data['lq_basename'], data['lq_path'], data['is_gray']):
        # define path of restored face image
        save_face_name = ''.join([lq_basename, opt['save_suffix'], '.', opt['helper']['save_ext']])
        save_restore_path = osp.join(opt['path']['results_path'], save_face_name)

        # clean all the intermediate results to process the next image
        face_helper.clean_all()
        
        lq_img = lq_img.unsqueeze(0)
        # read face image and get face landmarks to face helper
        if aligned: 
            # the input faces are already cropped and aligned
            face_helper.cropped_faces = [lq_img]
            face_helper.is_gray = is_gray.item()
        else:
            img_array = model._postprocess(lq_img)
            face_helper.read_image(img_array)

            if data_opt.get('pos_meta_root') is not None:
                # TODO: support get face landmarks from files
                # get face landmarks from meta position files
                face_helper.get_face_landmarks_5_from_file(data_opt['pos_meta_root'], lq_path)
            else:
                # get face landmarks for each face
                face_helper.get_face_landmarks_5(only_center_face=data_opt['only_center_face'], resize=640, eye_dist_threshold=5)

            face_helper.align_warp_face()
        
        # face restoration for each cropped face
        for idx, cropped_face_t in enumerate(face_helper.cropped_faces):
            with torch.no_grad():
                if not torch.is_tensor(cropped_face_t):
                    cropped_face_t = model._preprocess(cropped_face_t).unsqueeze(0)
                restored_face = model(cropped_face_t)
            face_helper.add_restored_face(restored_face)

        # paste face back to the image
        if not aligned:
            # upsample the background
            if bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = bg_upsampler.enhance(lq_img, outscale=opt['upscale'])[0]
            else:
                bg_img = None

            face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            restored_img = face_helper.paste_faces_to_input_image(
                save_path=save_restore_path, 
                upsample_img=bg_img
                )

            if restored_img is not None:
                # save the restored whole img if input not-aligned images
                imwrite(restored_img, save_restore_path)
                # results['restored_img'] = restored_img
                result = restored_img

        # save the orginal/restored cropped face images
        for idx, (cropped_face, restored_face) in enumerate(zip(face_helper.cropped_faces, face_helper.restored_faces)):
            # save cropped face
            # if opt['aligned']: 
                # save_crop_path = osp.join(opt['path']['results_path'], 'cropped_faces', f'{basename}_{idx:02d}.png')
                # imwrite(cropped_face, save_crop_path)

            # save restored face if input aligned face images
            if aligned:
                imwrite(restored_face, save_restore_path)
                result = restored_face # only a single face results when input aligned
            # else:
            #     save_face_name = f'{basename}_{idx:02d}.' + +opt['helper']['save_ext']

    # restored_face: float32 0-1
    # return {'restored_result': result, 'result_path': save_restore_path}

def inference_video_pipeline(data, model, opt, dataset_opt, device, logger):
    '''
    Args:
        data: a valid sequence to fed the video model (e.g. EDVR/BasiVSR)
    '''
    # seq_center_idx = int(dataset_opt['seq_length'] / 2)

    # seq shape as (B=1, T=seq_len, C, H, W)
    imgs_lq_seq = data['lq']
    with torch.no_grad():
        outputs = model(imgs_lq_seq)

    # loop the mini-batch to save restored frames
    for idx, (output, basename) in enumerate(zip(outputs, data['lq_basename'])):
        save_restore_path = osp.join(opt['path']['results_path'], 
                                    basename+opt['save_suffix']+'.'+opt['helper']['save_ext'])
        imwrite(output, save_restore_path)

    # save enhanced video
    # if dataset_opt['save_video_fps'] > 0:
    #     # TODO: support ffmpeg h264 decoding
    #     # write images to video
    #     h, w = restored_img[0].shape[:2]
        
    #     video_name += opt['save_suffix']
    #     save_restore_path = osp.join(opt['path']['results_path'], f'{video_name}.mp4')
    #     writer = cv2.VideoWriter(save_restore_path, cv2.VideoWriter_fourcc(*"mp4v"),
    #                             opt['save_video_fps'], (w, h))
    #     for f in restored_img:
    #         writer.write(f)
    #     writer.release()
    
    return {'restored_result': outputs, 'result_path': save_restore_path}