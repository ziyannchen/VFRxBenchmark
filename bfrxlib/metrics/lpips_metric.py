import argparse
import cv2
import glob
import numpy as np
import os.path as osp
from torchvision.transforms.functional import normalize
import torch
from tqdm import tqdm

from basicsr.utils import img2tensor
from bfrxlib.utils.registry import METRIC_REGISTRY

# https://download.pytorch.org/models/vgg16-397923af.pth
try:
    import lpips
except ImportError:
    print('Please install lpips: pip install lpips')


loss_fn_vgg = None
@METRIC_REGISTRY.register()
def calculate_lpips(data, device=None, **kargs):
    '''
    Params:
        data: dict, with keys ['img_restored'] and its data as cv2 image(WHC [0, 255] BGR uint8).
    '''
    # Configurations
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    global loss_fn_vgg
    if loss_fn_vgg is None:
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)  # RGB, normalized to [-1,1]

    img_restored = data['img_restored'].astype(np.float32) / 255.
    img_gt = data['img_gt'].astype(np.float32) / 255.

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    # FIXME: process the grey image
    if img_restored.ndim == 2:
        # print('Droping grey image... Continue')
        # continue
        print('Grey image... concating the only 1 channel to 3 channels')
        img_restored = np.concatenate((img_restored, img_restored, img_restored), axis=-1)

    img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
    # norm to [-1, 1]
    normalize(img_gt, mean, std, inplace=True)
    normalize(img_restored, mean, std, inplace=True)

    with torch.no_grad():
        # calculate lpips
        lpips_score = loss_fn_vgg(img_restored.unsqueeze(0).cuda(), img_gt.unsqueeze(0).to(device))
        # FIXME: to support cpu device
        lpips_score = lpips_score.cpu().item()

    return lpips_score
    

def calculate_lpips_folder(args, verbose=True):
    # Configurations
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()  # RGB, normalized to [-1,1]
    lpips_all = []

    img_list = sorted(glob.glob(osp.join(args.gt_folder, '*')))
    restored_list = sorted(glob.glob(osp.join(args.restored_folder, '*')))
    assert len(img_list) == len(restored_list), f'Error! The restored file length is not equal to GT file length \
        with GT({len(img_list)}) while restored({len(restored_list)}).'

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    for i, (restored_path, img_path) in enumerate(tqdm(zip(restored_list, img_list))):
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        img_restored = cv2.imread(restored_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        # FIXME: process the grey image
        if img_restored.ndim == 2:
            print('Droping grey image... Continue')
            continue
            print('Grey image... concating the only 1 channel to 3 channels')
            img_restored = np.concatenate((img_restored, img_restored, img_restored), axis=-1)
        # print(img_gt.shape, img_restored.shape)
        img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
        # norm to [-1, 1]
        normalize(img_gt, mean, std, inplace=True)
        normalize(img_restored, mean, std, inplace=True)

        # calculate lpips
        lpips_val = loss_fn_vgg(img_restored.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda())
        lpips_val = lpips_val.cpu().item()
        # print(f'{i+1:3d}: {basename:25}. \tLPIPS: {lpips_val:.6f}.')
        lpips_all.append(lpips_val)
    
    lpips_result = sum(lpips_all) / len(lpips_all)
    
    if verbose:
        print(args.gt_folder)
        print(args.restored_folder)
        print(f'Average: LPIPS: {lpips_result:.6f}')
    return lpips_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-restored_folder', type=str, help='Path to the folder.', required=True)
    parser.add_argument('-gt_folder', type=str, help='Path to the folder.', required=True)
    args = parser.parse_args()
    calculate_lpips(args)
