import cv2
import math
import numpy as np
import os.path as osp
import torch
from torchvision.utils import make_grid
from PIL import Image
from torchvision.transforms.functional import normalize

from basicsr.utils import img2tensor, scandir

def is_gray(img, threshold=5):
    """
    img: [0, 255] numpy.ndarray
    """
    img = Image.fromarray(img.astype(np.uint8))
    if len(img.getbands()) == 1:
        return True
    img1 = np.asarray(img.getchannel(channel=0), dtype=np.int16)
    img2 = np.asarray(img.getchannel(channel=1), dtype=np.int16)
    img3 = np.asarray(img.getchannel(channel=2), dtype=np.int16)
    diff1 = (img1 - img2).var()
    diff2 = (img2 - img3).var()
    diff3 = (img3 - img1).var()
    diff_sum = (diff1 + diff2 + diff3) / 3.0
    if diff_sum <= threshold:
        return True
    else:
        return False

def rgb2gray(img, out_channel=3):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    if out_channel == 3:
        gray = gray[:,:,np.newaxis].repeat(3, axis=2)
    return gray

def bgr2gray(img, out_channel=3):
    b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    if out_channel == 3:
        gray = gray[:,:,np.newaxis].repeat(3, axis=2)
    return gray

def read_img_seq(path, return_imgname=False, transform_fn=None, judge_gray=False):
    """Read a sequence of images from a given folder path.

    Args:
        path (list[str] | str): List of image paths or image folder path.
        require_mod_crop (bool): Require mod crop for each image.
            Default: False.
        scale (int): Scale factor for mod_crop. Default: 1.
        return_imgname(bool): Whether return image names. Default False.

    Returns:
        Tensor: size (t, c, h, w), RGB, [0, 1].
        list[str]: Returned image name list.
    """
    if isinstance(path, list):
        img_paths = path
    else:
        img_paths = sorted(list(scandir(path, full_path=True)))
    
    res = {}
    imgs = []
    if transform_fn is not None:
        for v in img_paths:
            img = cv2.imread(v).astype(np.float32)
            imgs.append(transform_fn(img))
        # imgs = [normalize(img, norm_params['mean'], norm_params['std'], inplace=True) for img in imgs]
    else:
        for v in img_paths:
            img = cv2.imread(v).astype(np.float32)
            imgs.append(img / 255.)
        imgs = img2tensor(imgs, bgr2rgb=True, float32=True)

    if judge_gray:
        is_grayscale = is_gray(img)
        res['is_gray'] = is_grayscale
    imgs = torch.stack(imgs, dim=0)
    res['imgs'] = imgs

    if return_imgname:
        imgnames = [osp.splitext(osp.basename(path))[0] for path in img_paths]
        res['imgnames'] = imgnames
    return res
