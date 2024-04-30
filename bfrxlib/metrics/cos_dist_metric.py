import argparse
import cv2
import glob
import math
import numpy as np
import os
import torch
from tqdm import tqdm
# from .arcface.config.config import Config
# from .arcface.models.resnet import resnet_face18
from torch.nn import DataParallel
from torch.nn import functional as F
from torchvision.transforms.functional import normalize

from basicsr.utils import img2tensor
from bfrxlib.utils.registry import METRIC_REGISTRY
from bfrxlib.metrics.basic import cal_distance


cos_dis_model_name = None
cos_dis_model = None
def init_model(backbone, device=None):
    '''
    Args:
        backbone: str, the backbone of the arcface model, options: ['arcface_resnet18', 'arcface_ir_se50'].
    '''
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    global cos_dis_model
    global cos_dis_model_name
    if cos_dis_model is None or cos_dis_model_name != backbone:
        cos_dis_model_name = backbone
        print('using arcface backbone', backbone)
        from bfrxlib.preprocess.models import Descriptor
        cos_dis_model = Descriptor(backbone, device=device)
        
    return cos_dis_model

def calculate_cos_dist(img, img_gt, device=None, backbone='arcface_resnet18'):
    '''
    Params:
        img: np.ndarray, cv2 image(WHC [0, 255] BGR uint8).
        img_gt: np.ndraay, cv2 image(WHC [0, 255] BGR uint8).
    '''
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cos_dis_model = init_model(backbone=backbone, device=device)
    img = cos_dis_model._preprocess(img)
    img_gt = cos_dis_model._preprocess(img_gt)

    data = torch.cat([img, img_gt], dim=0)
    # data = data.to(device)
    output = cos_dis_model(data)
    output = output.data.cpu().numpy() if device.type == 'cuda' else output.data.numpy()
    dist = cal_distance(output[0], output[1], dis_type='cosine')
    
    return dist


@METRIC_REGISTRY.register()
def calculate_idd(data, device=None, backbone='arcface_resnet18', **kargs):
    '''IDD(ID Distance): 
        The cosine distance of the [restored image] and the [gt image] 
        in embedding space based on arcface.

    Args:
        data: dict, with keys ['img_restored', 'img_gt'] and its respect data as cv2 image(WHC [0, 255] BGR uint8).
        backbone: str, the backbone of the arcface model, default: 'arcface_resnet18'.
    '''
    img = data['img_restored']
    img_gt = data['img_gt']
    id_dist = calculate_cos_dist(img, img_gt, device=device, backbone=backbone)

    return id_dist

@METRIC_REGISTRY.register()
def calculate_ids(data, device=None, backbone='arcface_resnet18', **kargs):
    '''IDS(ID Similarity: ref to gt): 
        The cosine similarity of the [restored image] and the [gt image] 
        in embedding space based on arcface.
            p.s. IDS equals 1-IDD.   
    Args:
        data: dict, with keys ['img_restored', 'img_gt'] and its respect data as cv2 image(WHC [0, 255] BGR uint8).
        backbone: str, the backbone of the arcface model, default: 'resnet18'.
    '''
    id_dist = calculate_idd(data, device, backbone)

    return 1-id_dist

@METRIC_REGISTRY.register()
def calculate_idp(data, device=None, backbone='resnet18', **kargs):
    '''IDP(ID Similarity: ref to lq): 
        The cosine similarity of the [restored image] and the [lq image] 
        in embedding space based on arcface.

    Args:
        data: dict, with keys ['img_restored', 'img_lq'] and its respect data as cv2 image(WHC [0, 255] BGR uint8).
        backbone: str, the backbone of the arcface model, default: 'resnet18'.
    '''
    img = data['img_restored']
    img_ref = data['img_lq']
    id_dist = calculate_cos_dist(img, img_ref, device=device, backbone=backbone)

    return id_dist

@METRIC_REGISTRY.register()
def calculate_vidd(frame_list, device=None, backbone='arcface_resnet18', distance_type='l2', **kargs):
    '''VIDD(Video ID Distance): 
        The average cosine similarity between every [restored frame] and its [neighbouring frame] 
        in embedding space based on Arcface.

    Args:
        data: dict, with keys ['img_restored'] and its respect data as cv2 image(WHC [0, 255] BGR uint8).
            data['img_restored'] should be a list of cv2 images from a video clip.
        backbone: str, the backbone of the arcface model, default: 'arcface_resnet18'.
    '''
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cos_dis_model = init_model(backbone=backbone, device=device)

    feat_matrix = []
    for frame in frame_list:
        frame = cos_dis_model._preprocess(frame)
        output = cos_dis_model(frame)
        feat_matrix.append(output)
    feat_matrix_ori = torch.cat(feat_matrix, dim=0)
    first = feat_matrix.pop(0)
    feat_matrix.append(first)
    feat_matrix_shift = torch.cat(feat_matrix, dim=0)
    diff = cal_distance(feat_matrix_ori, feat_matrix_shift, dis_type=distance_type)
    diff = diff[:-1]
    # print(diff)

    vidd_score = sum(diff) / len(diff)
    return vidd_score.data.cpu().numpy() if device.type == 'cuda' else output.data.numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-restored_folder', type=str, help='Path to the folder.', required=True)
    parser.add_argument('-gt_folder', type=str, help='Path to the folder.', required=True)
    parser.add_argument('-test_model_path', type=str, default='experiments/pretrained_models/metric_weights/resnet18_110.pth')
    args = parser.parse_args()
    calculate_cos_dist(args)
