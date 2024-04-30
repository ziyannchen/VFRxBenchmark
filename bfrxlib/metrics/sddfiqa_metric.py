import os
import cv2
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as opti
import torchvision.transforms as T

import numpy as np
from scipy import stats
import pdb
from PIL import Image

from bfrxlib.utils.registry import METRIC_REGISTRY
from bfrxlib.utils import align_face
from .sdd_fiqa import model

def read_img(imgPath, data_type):     # read image & data pre-process
    data = torch.randn(1, 3, 112, 112)
    
    
    data[0, :, :, :] = transform(img)
    return data

def network(eval_model, device):
    net = model.R50([112, 112], use_type="Qua").to(device)
    net_dict = net.state_dict()     
    data_dict = {
        key.replace('module.', ''): value for key, value in torch.load(eval_model, map_location=device).items()}
    net_dict.update(data_dict)
    net.load_state_dict(net_dict)
    net.eval()
    return net


sdd_fiqa_model = None
@METRIC_REGISTRY.register()
def calculate_sddfiqa(data, device=None, **kargs):
    """Modified from sdd_fiqa/quality/eval.py
    """
    img = data['img_restored']
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    global sdd_fiqa_model
    if sdd_fiqa_model is None:
        METRICS_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        eval_model = osp.join(METRICS_ROOT_DIR, 'sdd_fiqa/weights/SDD_FIQA_checkpoints_r50.pth')
        sdd_fiqa_model = network(eval_model, device)
    
    transform = T.Compose([
        # T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    # img = Image.open(imgPath).convert("RGB")
    img = align_face(img, target_face="mtcnn", outsize=(112, 112), landmarks='ffhq') # HWC, BGR, [0, 255]
    
    input_data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img[..., ::-1]
    input_data = transform(input_data).to(device).unsqueeze(0) # HWC, RGB, [0, 255]
    # calculate score
    sddfiqa_score = float(sdd_fiqa_model(input_data).data.cpu().numpy().squeeze())

    return sddfiqa_score
