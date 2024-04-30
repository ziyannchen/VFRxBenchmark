import numpy as np
import torch
import cv2

from facexlib.utils.misc import img2tensor
from torchvision.transforms.functional import normalize

def to_gray(img, mode='gray', retain_dim=True):
    '''
    Args:
        img: [C, H, W] torch.Tensor or [H, W, C] cv2.numpy
    '''
    weight = {
        # 'r': 0.2989,
        'r': 0.299,
        'g': 0.5870,
        'b': 0.1140
    }
    if type(img) == np.ndarray:
        getChannel = lambda i: img[:, :, i] # slice the channel dim
        expChannel = lambda img: img[..., np.newaxis] # retain the channel dim
    elif type(img) == torch.Tensor:
        getChannel = lambda i: img[i, :, :]
        expChannel = lambda img: img.unsqueeze(0)

    if mode == 'gray':
        image = img
    elif mode == 'bgr':
        image = weight['b'] * getChannel(0) + weight['g'] * getChannel(1) + weight['r'] * getChannel(2)
    elif mode == 'rgb':
        # for rgb images
        image = weight['r'] * getChannel(0) + weight['g'] * getChannel(1) + weight['b'] * getChannel(2)
    else:
        raise NotImplementedError(f'Unsupported mode {mode} for preprocess.')
    
    if retain_dim:
        image = expChannel(image)
    return image

class Descriptor:
    def __init__(self, net_name='arcface_ir_se50', device=None) -> None:
        '''
        Face image descriptor. Ref: facexlib.
            (currently only support ArcFace with differeny backbones)
        Args:
            net_name (str, optional): The name of the face recognition model. Defaults to 'arcface'. 
                Options: ['arcface_ir_se50', 'arcface_resnet18'].
        '''
        from . import init_recognition_model
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.input_size = {
            'arcface_ir_se50': 112,
            'arcface_resnet18': 128
        }[net_name]
        self.net_name = net_name
        self.recog_net = init_recognition_model(net_name, device=device)

    def _preprocess(self, img):
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        img_t = img2tensor(img / 255., bgr2rgb=True, float32=True)
        if self.net_name == 'arcface_resnet18':
            img_t = to_gray(img_t, mode='rgb')
        else:
            normalize(img_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        img_t = img_t.unsqueeze(0).to(self.device)
        return img_t

    def __call__(self, img_t):
        with torch.no_grad():
            output = self.recog_net(img_t)
        return output
    
