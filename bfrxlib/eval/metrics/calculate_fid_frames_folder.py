import argparse
import math
import numpy as np
import os
import torch
import shutil
from torch.utils.data import DataLoader
import glob
from tqdm import tqdm

from vqfr.data import build_dataset
from vqfr.metrics.fid import calculate_fid, extract_inception_features, load_patched_inception_v3

from .calculate_fid_folder import calculate_fid_folder
from bfrxlib.preprocess.utils import makedir

class Args:
    def __init__(self, restored_folder, fid_stats, batch_size=64, num_sample=3000, num_workers=4, backend='disk') -> None:
        """_summary_

        Args:
            restored_folder (_type_): _description_
            fid_stats (_type_): Path to the dataset fid statistics.
            batch_size (int, optional): _description_. Defaults to 64.
            num_sample (int, optional): _description_. Defaults to 3000.
            num_workers (int, optional): _description_. Defaults to 4.
            backend (str, optional): io backend for dataset. Option: disk, lmdb. Defaults to 'disk'.
        """
        self.restored_folder = restored_folder
        self.fid_stats = fid_stats
        self.batch_size = batch_size
        self.num_sample = num_sample
        self.num_workers = num_workers
        self.backend = backend

def main(args):
    print(args.restored_frames_folder)
    all_dir = os.listdir(args.restored_frames_folder)
    print(len(all_dir), 'frames folders')
    
    tmp_link_dir = '_'.join(args.restored_frames_folder.split('/')[-2:])+'fid_frames_folder_tmp'
    makedir(tmp_link_dir, rebuild=True)
    pbar = tqdm(total=len(all_dir), desc='Generate Softlinks', unit='files')
    for dir in all_dir:
        pbar.update(1)
        input_dir = os.path.join(args.restored_frames_folder, dir)
        all_ims = glob.glob(os.path.join(input_dir, '*.png'))
        # all_ims = glob.glob(os.path.join(input_dir, 'final_results', '*.png'))
        for im in all_ims:
            os.symlink(im, os.path.join(tmp_link_dir, dir+'_'+os.path.basename(im)))
    args_tmp = Args(tmp_link_dir, args.fid_stats)
    fid = calculate_fid_folder(args_tmp, verbose=True)
    print('Average FID: ', fid)
    shutil.rmtree(tmp_link_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--restored_frames_folder', type=str, help='Path to the folder.', required=True)
    parser.add_argument(
        '--fid_stats',
        type=str,
        help='Path to the dataset fid statistics.',
        default='experiments/pretrained_models/metric_weights/inception_FFHQ_512.pth')
    args = parser.parse_args()
    main(args)
