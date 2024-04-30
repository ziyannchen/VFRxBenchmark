import torch
import numpy as np
import os
import cv2

from .serfiq.face_image_quality import SER_FIQ

from bfrxlib.utils import align_face
from bfrxlib.utils.registry import METRIC_REGISTRY

model = None
@METRIC_REGISTRY.register()
def calculate_serfiq(data, device=None):
    '''
    Params:
        data: dict, with keys ['img_restored'] and its data as cv2 image(WHC [0, 255] BGR uint8).
            Attention: the image should be aligned and cropped as the FFHQ face.
    '''
    img = data['img_restored']
    
    global model
    if model is None:
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SER_FIQ(gpu=0 if device == 'cuda' else None)

    aligned_img = align_face(img, target_face='mtcnn', outsize=(112, 112), landmarks='ffhq')
    aligned_img = aligned_img[..., ::-1].transpose(2, 0, 1) # RGB, CHW, [0, 255]

    # Calculate the quality score of the image
    # T=100 (default) is a good choice
    # Alpha and r parameters can be used to scale your score distribution.
    with torch.no_grad():
        serfiq_score = model.get_score(aligned_img, T=100)

    return serfiq_score