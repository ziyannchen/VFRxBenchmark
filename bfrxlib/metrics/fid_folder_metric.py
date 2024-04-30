import argparse
import math
import numpy as np
import os
import torch
from torch.utils.data import DataLoader

from basicsr.data import build_dataset
from basicsr.metrics.fid import calculate_fid, extract_inception_features, load_patched_inception_v3
from bfrxlib.utils.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
def calculate_fid_folder(restored_folder, fid_stats='inception_FFHQ_512', device=None, 
            backend='disk', mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5], num_workers=4, num_sample=100000, batch_size=64, verbose=True, **kargs):
    '''
        restored_folder: str or list. To point location of the restored images.
        fid_stats: The reference fis statistics. inception_FFHQ_512.pth as default.
        num_sample: The maximum number of sampled data to use.
    '''
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # inception model
    inception = load_patched_inception_v3(device)
    print(f'Using {fid_stats}.pth to calculate fid.')
    # create dataset
    opt = {}
    opt['name'] = 'SingleImageDataset'
    opt['type'] = 'SingleImageDataset'
    opt['dataroot_lq'] = restored_folder
    opt['io_backend'] = dict(type=backend)
    opt['mean'] = mean
    opt['std'] = std
    
    if isinstance(restored_folder, list):
        with open('fid_folder_tmp.txt', 'w') as f:
            f.write('\n'.join(restored_folder))
        opt['dataroot_lq'] = ''
        opt['meta_info_file'] = 'fid_folder_tmp.txt'

    dataset = build_dataset(opt)

    # create dataloader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=None,
        drop_last=False)
    if verbose:
        print(len(data_loader))

    # FIXME: why need sampling? why not use all data?
    num_sample = min(num_sample, len(dataset))
    total_batch = math.ceil(num_sample / batch_size)

    def data_generator(data_loader, total_batch):
        for idx, data in enumerate(data_loader):
            if idx >= total_batch:
                break
            else:
                yield data['lq']

    features = extract_inception_features(data_generator(data_loader, total_batch), inception, total_batch, device)
    features = features.numpy()
    total_len = features.shape[0]
    features = features[:num_sample]

    if verbose:
        print(f'Extracted {total_len} features, use the first {features.shape[0]} features to calculate stats.')

    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    # load the dataset stats
    METRICS_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    fid_stats = os.path.join(METRICS_ROOT_DIR, 'fid', fid_stats+'.pth')
    print('fid_stat_path: ', fid_stats)
    stats = torch.load(fid_stats)
    real_mean = stats['mean']
    real_cov = stats['cov']

    # calculate FID metric
    fid = calculate_fid(sample_mean, sample_cov, real_mean, real_cov)

    if verbose:
        # print(restored_folder)
        print('fid:', fid)
    return fid


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-restored_folder', type=str, help='Path to the folder.', required=True)
    parser.add_argument(
        '--fid_stats',
        type=str,
        help='Path to the dataset fid statistics.',
        default='weights/stats/inception_FFHQ_512.pth')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_sample', type=int, default=10000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--backend', type=str, default='disk', help='io backend for dataset. Option: disk, lmdb')
    args = parser.parse_args()
    calculate_fid_folder(**args)
