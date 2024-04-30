import os
import pyiqa
import torch
import numpy as np

from bfrxlib.utils.registry import METRIC_REGISTRY

# class BRISQUE:
#     def __init__(self, device=None):
#         if device is not None:
#             self.device = device
#         else:
#             self.device = 'cuda' if torch.cuda_is_available() else 'cpu'
#         self.net = pyiqa.create_metric('brisque', device=device, as_loss=False)

#     def __call__(self, img):
#         """
#         img: torch.tensor BCHW [0, 1]
#             with the normalization process from cv2.imread Numpy.array as follows:
#             img = img[..., ::-1].copy() # RGB, HWC, [0, 255]
            
#             # (N, C, H, W), RGB, [0, 1]
#             img = np.array(img).transpose(2, 0, 1) / 255
#             img = torch.tensor(img[None, ...]).float().to(device)
#         """
#         score = float(metric(img).detach().cpu().numpy()[0])

#         return score

brisque_net = None
@METRIC_REGISTRY.register()
def calculate_brisque(data, device=None, **kwargs):
    '''
    Params:
        data: dict, with keys ['img_restored'] and its data as cv2 image(WHC [0, 255] BGR uint8).
    '''
    img = data['img_restored']
    
    global brisque_net
    if brisque_net is None:
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        brisque_net = pyiqa.create_metric('brisque', device=device, as_loss=False)
    img = img[..., ::-1].copy() # RGB, HWC, [0, 255]
        
    # (N, C, H, W), RGB, [0, 1]
    img = np.array(img).transpose(2, 0, 1) / 255
    img = torch.tensor(img[None, ...]).float().to(device)
    # print(type(score))- # torch.Tensor, size = (1, 1)
    # FIXME: to support cpu device
    with torch.no_grad():
        score = float(brisque_net(img).detach().cpu().numpy()[0])

    return score
