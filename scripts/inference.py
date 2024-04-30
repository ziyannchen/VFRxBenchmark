import logging
import torch
from copy import deepcopy
from tqdm import tqdm
import os
from os import path as osp

from basicsr.utils import img2tensor, tensor2img, get_time_str

from bfrxlib.data import build_dataset, build_dataloader
from bfrxlib.utils import is_gray, dict2str, get_root_logger, get_env_info, yaml_find
from bfrxlib.models import build_model
from bfrxlib.metrics import calculate_metric
from bfrxlib.preprocess.utils.video import images2video

from inference_pipelines import inference_image_pipeline, inference_video_pipeline, parse_options

import warnings
warnings.filterwarnings("ignore")

def main(root_path):
    # inference pipeline
    opt, args = parse_options(root_path)
    device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')

    # ------------------ set up logger ----------------------
    # log_file = osp.join(opt['path']['logs_root'], f"test_{opt['name']}_{get_time_str()}.log")
    log_file = osp.join(opt['path']['logs_root'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name=opt['name'], log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # ------------------ set up the target model ----------------------
    model = build_model(opt)

    # ------------------ set up dataloaders ----------------------
    dataset_opt = opt['dataset']
    test_set = build_dataset(dataset_opt, dict(transform_fn=model._preprocess))
    dataloader = build_dataloader(
        test_set, dataset_opt, sampler=None)
    logger.info(f"Total number of targets in {dataset_opt['name']} test set: {len(test_set)}.")

    # -------------------- inference ---------------------
    pbar = tqdm(total=len(dataloader), unit='ims')
    for data in dataloader:
        pbar.update(1)
        pbar.set_description('Processing '+data['lq_path'][0][-50:])

        # restoration inference
        if 'video' in model.inference_type:
            # for vsr models
            inference_video_pipeline(data, model, opt, dataset_opt, device, logger)
        else:
            # for single image restoration models
            inference_image_pipeline(data, model, opt, device)

    if dataset_opt.get('save_video_fps', 0):
        save_path = opt['path']['results_path'] + '_mp4'
        os.makedirs(save_path, exist_ok=True)
        restored_clip_dir = os.listdir(opt['path']['results_path'])
        for clip_dir in restored_clip_dir:
            images2video(osp.join(opt['path']['results_path'], clip_dir), osp.join(save_path, clip_dir+'.mp4'), fps=dataset_opt.get('save_video_fps', 25), ffmpeg=True, suffix='png')

    logger.info('\nAll restored image results have saved in '+opt['path']['results_path'])

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    main(root_path)