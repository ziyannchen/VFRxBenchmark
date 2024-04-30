import logging
import torch
from copy import deepcopy
from tqdm import tqdm
from os import path as osp

from basicsr.utils import img2tensor, tensor2img, get_time_str

from bfrxlib.data import build_dataset, build_dataloader
from bfrxlib.utils import is_gray, dict2str, get_root_logger, get_env_info, FaceRestoreHelper, yaml_find
from bfrxlib.models import build_model
from bfrxlib.metrics import calculate_metric

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
    test_set = build_dataset(dataset_opt, dict(transform_fn=model._preprocess))
    dataloader = build_dataloader(
        test_set, dataset_opt, sampler=None)
    logger.info(f"Total number of targets in {dataset_opt['name']} test set: {len(test_set)}.")

    # -------------------- start to processing ---------------------
    with_metrics = dataset_opt.get('metrics') is not None
    if with_metrics:
        metric_results = dict()  # {metric: 0 for metric in dataset_opt['metrics'].keys()}
        if 'fid_folder' in dataset_opt['metrics']:
            # fid should be calculated on the distribution scale
            cal_fid_opt = yaml_find(opt, 'val.metrics.fid_folder')
            cal_fid_opt = cal_fid_opt if cal_fid_opt is not None else dict(type='calculate_fid_folder')
            dataset_opt['metrics'].remove('fid_folder')
        for target_metric in dataset_opt['metrics']:
            metric_results[target_metric] = []

    opt_ = deepcopy(opt)
    opt_['path']['results_root'] = osp.join(opt_['path']['results_root'], dataset_opt['name'])

    pbar = tqdm(total=len(dataloader), unit='ims')
    for data in dataloader:
        pbar.update(1)
        pbar.set_description('Processing '+data['lq_path'][0])

        # restoration inference
        if 'video' in model.inference_type:
            # for vsr models
            outputs = inference_video_pipeline(data, model, opt_, dataset_opt, device, logger)
        else:
            # ------------------ set up FaceRestoreHelper -------------------
            face_helper = FaceRestoreHelper(device=device, face_size=dataset_opt['face_size'], **opt['helper'])
            # for single image restoration models
            outputs = inference_image_pipeline(data, model, face_helper, opt_, dataloader, device)

        # calculate evaluation metrics
        if with_metrics:
            # TODO: speed up the evaluation by supporting all-tensor workflow
            eval_data = {'img_restored': outputs['restored_result'], 'device': device}
            if 'gt' in data:
                eval_data['img_gt'] = model._postprocess(data['gt'].squeeze().unsqueeze(0))

            # calculate metrics
            for target_metric in dataset_opt['metrics']:
                if yaml_find(opt, f'val.metrics.{target_metric}') is not None:
                    metric_opt = opt['val']['metrics'][target_metric]
                else:
                    metric_opt = dict(type=f'calculate_{target_metric}')
                print(target_metric)
                if target_metric in ['idp']:
                    print('idp')
                    eval_data.update({'img_lq': model._postprocess(data['lq'].squeeze().unsqueeze(0))})

                metric_results[target_metric].append(calculate_metric(dict(data=eval_data), metric_opt))

        del outputs
        torch.cuda.empty_cache()

    if with_metrics:
        if cal_fid_opt is not None:
            metric_results['fid_folder'] = calculate_metric(dict(restored_folder=opt_['path']['results_root']), cal_fid_opt)

        for target_metric in dataset_opt['metrics']:
            metric_results[target_metric] = sum(metric_results[target_metric]) / len(metric_results[target_metric])
    logger.critical(opt['name']+' '+dataset_opt['name']+'\n'+dict2str(metric_results))
    logger.info('\nAll restored image results have saved in '+opt['path']['results_root'])

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    main(root_path)