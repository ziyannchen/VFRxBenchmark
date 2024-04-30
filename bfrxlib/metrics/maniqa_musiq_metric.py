import os
import pyiqa
import torch
from tqdm import tqdm
import numpy as np
import time

from bfrxlib.utils.registry import METRIC_REGISTRY

musiq_model = None
@METRIC_REGISTRY.register()
def calculate_musiq(data, device=None, **kargs):
    """
    data: {'restored_img': img # BGR, HWC, [0, 255]}
    """
    img = data['img_restored']
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    global musiq_model
    if musiq_model is None:
        # TODO: to load model only once can run 10 times faster
        musiq_model = pyiqa.create_metric('musiq', device=device, as_loss=False)
    
    # (N, C, H, W), RGB, [0, 1]
    img = np.array(img[..., ::-1]).transpose(2, 0, 1) / 255
    img = torch.tensor(img[None, ...]).float().to(device)
    # calculate score
    score = musiq_model(img) # torch.Tensor, size = (1, 1)
    # score = float(score.detach().cpu().numpy()[0][0])
    musiq_score = float(score.detach().cpu().numpy())

    return musiq_score

maniqa_model = None
@METRIC_REGISTRY.register()
def calculate_maniqa(data, device=None, **kargs):
    """
    data: {'restored_img': img # BGR, HWC, [0, 255]}
    """
    img = data['img_restored']
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    global maniqa_model
    if maniqa_model is None:
        maniqa_model = pyiqa.create_metric('maniqa', device=device, as_loss=False)
    # metric = pyiqa.create_metric('maniqa', device=device, as_loss=False)
    
    img = img[..., ::-1].copy() # RGB, HWC, [0, 255]
    
    # (N, C, H, W), RGB, [0, 1]
    img = np.array(img).transpose(2, 0, 1) / 255
    img = torch.tensor(img[None, ...]).float().to(device)
    # calculate score
    # print(type(score)) # torch.Tensor, size = (1, 1)
    maniqa_score = float(maniqa_model(img).detach().cpu().numpy()[0][0])

    return maniqa_score