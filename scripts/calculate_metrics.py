import glob
import cv2
import os
import logging
from tqdm import tqdm
from os import path as osp

from basicsr.utils import get_time_str, scandir
from bfrxlib.metrics import calculate_metric
from bfrxlib.utils import dict2str, get_root_logger, get_env_info

import warnings
warnings.filterwarnings("ignore")



def main(args):
    os.makedirs(osp.dirname(args.log_file), exist_ok=True)
    logger = get_root_logger(logger_name='bfrxlib_cal_metrics', log_level=logging.INFO, log_file=args.log_file)
    logger.info(get_env_info())

    legal_metrics = [
        # full-reference
        'psnr', 'ssim', 'lpips', 
        'ids', 'idd', 
        # no-reference
        'niqe', 'brisque', 'maniqa', 'musiq', 'hyperiqa', 
        'idp', 'vidd',
        'fid_folder', 'fid_folder_celeba',
        'serfiq', 'sddfiqa', 'ifqa', 'faceqnet'
        ]

    need_refer = ['psnr', 'ssim', 'lpips', 'ids']
    need_lq = ['idp']
    opt_default = {
        'hyperiqa': {
            'type': 'calculate_hyperiqa',
            'crop_border': 50
        },
        'fid_folder': {
            'type': 'calculate_fid_folder',
            'fid_stats': 'inception_FFHQ_512', # options: ['inception_FFHQ_512', 'inception_CelebA_512']
        },
        'fid_folder_celeba': {
            'type': 'calculate_fid_folder',
            'fid_stats': 'inception_CelebA_512', # options: ['inception_FFHQ_512', 'inception_CelebA_512']
        },
        'vidd': {
            'type': 'calculate_vidd',
            'backbone': 'arcface_resnet18',
            'distance_type': 'l2'
        }
    }
    metrics_opt = {}
    metric_results = {}
    for metric in args.metrics:
        assert metric in legal_metrics, f'Unsupported metrics {metric}, supported {str(legal_metrics)}.'
        if metric in need_refer:
            assert args.gt_folder is not None, f'GT folder is required for {metric} calculation.'
        if metric in need_lq:
            assert args.lq_folder is not None, f'LQ folder is required for {metric} calculation.'

        if metric not in opt_default:
            metrics_opt.update({metric: dict(type=f'calculate_{metric}')})
        else:
            metrics_opt[metric] = opt_default[metric]
        metric_results[metric] = []
    
    logger.info('To calculate metrics: '+str(args.metrics))
    # restored image
    all_img_paths = sorted(list(scandir(args.restored_folder, suffix=('.jpg', '.png'), recursive=True, full_path=True)))
    # all_imgs = sorted([i for i in all_imgs if 'restored_faces' in i])
    all_imgs = [cv2.imread(i) for i in all_img_paths]
    logger.info('Input restored folder: ' + args.restored_folder)

    def fid_folder(matric_name):
        # FID should be calculated on the scale of a data distribution
        cal_fid_opt = metrics_opt.pop(matric_name)
        fid_score = calculate_metric(dict(restored_folder=all_img_paths), cal_fid_opt)
        metric_results[matric_name] = fid_score
        logger.info(dict2str(cal_fid_opt))

    if 'fid_folder' in metrics_opt:
        fid_folder('fid_folder')
    if 'fid_folder_celeba' in metrics_opt:
        fid_folder('fid_folder_celeba')
    if 'vidd' in metrics_opt:
        vidd_opt = metrics_opt.pop('vidd')
        # FIXME: not finished yet
        all_clip_dir = os.listdir(args.restored_folder)
        vidd_scores = []
        for clip_dir in all_clip_dir:
            frame_list = []
            for idx, path in enumerate(all_img_paths):
                if clip_dir in path:
                    frame_list.append(all_imgs[idx])
            vidd_score = calculate_metric(dict(frame_list=frame_list), vidd_opt)
            vidd_scores.append(float(vidd_score))
        metric_results['vidd'] = sum(vidd_scores) / len(vidd_scores)
    
    if args.lq_folder is not None:
        all_imgs_lq = sorted(scandir(args.lq_folder, suffix=('.jpg', '.png'), recursive=True, full_path=True))
        assert len(all_imgs_lq) == len(all_imgs), f'len(lq)={len(all_imgs_lq)} not equals to len(estored)={len(all_imgs)}.'
    if args.gt_folder is not None:
        all_imgs_gt = sorted(scandir(args.gt_folder, suffix=('.jpg', '.png'), recursive=True, full_path=True))
        assert len(all_imgs_gt) >= len(all_imgs), f'len(gt)={len(all_imgs_gt)} not equals to len(estored)={len(all_imgs)}.'
    logger.info('Total image number: ' + str(len(all_imgs)))
    
    assert len(all_imgs) != 0, f'No images found in the input folder {args.restored_folder}.'
    
    pbar = tqdm(total=len(all_imgs))
    for idx, (img, img_path) in enumerate(zip(all_imgs, all_img_paths)):
        pbar.update(1)
        pbar.set_description(f'Processing {os.path.basename(img_path)}')
        for target_metric, metric_opt in metrics_opt.items():
            # img = cv2.imread(img_path)
            data = {'img_restored': img}
            if target_metric in need_refer:
                img_gt = cv2.imread(all_imgs_gt[idx])
                data['img_gt'] = img_gt
            if target_metric in need_lq:
                img_lq = cv2.imread(all_imgs_lq[idx])
                data['img_lq'] = img_lq
            score = calculate_metric(dict(data=data), metric_opt)
            metric_results[target_metric].append(score)
    
    for target_metric in metrics_opt:
        scores_tmp = metric_results[target_metric]
        metric_results[target_metric] = sum(scores_tmp) / len(scores_tmp)

    logger.critical(f'Path: {args.restored_folder}. \tResults: '+'\n'+dict2str(metric_results))
    return metric_results


if __name__ == '__main__':
    import pandas as pd
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--restored_folder', type=str, required=True)
    parser.add_argument('-gt', '--gt_folder', type=str, help='Folder of the ground truth counterparts.')
    parser.add_argument('-lq', '--lq_folder', type=str, help='Folder of the low-quality counterparts.')
    parser.add_argument('-m', '--metrics', nargs='+', help='Option: [psnr, ssim, lpips, brisque, hyperiqa, maniqa, musiq, niqe, fid, serfiq, sdd_fiqa, ifqa, faceqnet]')
    parser.add_argument('-log', '--log_file', type=str, default=f'logs/test/test_tmp_{get_time_str()}.log')
    args = parser.parse_args()

    res = main(args)

    # save the results to a .csv file
    PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_file = f'{PROJ_DIR}/'+args.log_file.replace('.log', '')+'.csv'
    res['datapath'] = [args.restored_folder]
    for key in res:
        res[key] = [res[key]]
    pd.DataFrame.from_dict(res).to_csv(csv_file, index=False)
    print(f'DataFrame of the results saved to {csv_file}')