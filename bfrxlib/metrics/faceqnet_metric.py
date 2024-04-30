from keras.models import load_model
import numpy as np
import os
import cv2

from bfrxlib.utils import align_face
from bfrxlib.utils.registry import METRIC_REGISTRY

model = None
@METRIC_REGISTRY.register()
def calculate_faceqnet(data, device=None):
    img = data['img_restored']
    
    METRICS_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    faceqnet_ckpt = os.path.join(METRICS_ROOT_DIR, 'faceqnet', 'FaceQnet_v1.h5')
    global model
    if model is None:
        model = load_model(faceqnet_ckpt)
    
    aligned_img = align_face(img, target_face='mtcnn', outsize=(224, 224), landmarks='ffhq').astype('float32')[None, ...]
    faceqnet_score = model.predict(aligned_img, batch_size=1, verbose=0).squeeze()

    return faceqnet_score