import os
import cv2
import torch
import numpy as np
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

from torchvision.transforms.functional import normalize
from bfrxlib.utils.registry import METRIC_REGISTRY
from basicsr.utils import img2tensor
from .ifqa.model import Discriminator


# transform_input = A.Compose([
#     A.Resize(256, 256, interpolation=cv2.INTER_LINEAR),
#     A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
#     ToTensorV2()
# ])

def get_model(device):
    discriminator = Discriminator().to(device)
    METRICS_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    model_file_path = f"{METRICS_ROOT_DIR}/ifqa/weights/IFQA_Metric.pth"
    discriminator.load_state_dict(torch.load(model_file_path, map_location=device)['D'])
    discriminator.eval()
    return discriminator

model = None
@METRIC_REGISTRY.register()
def calculate_ifqa(data, device=None):
    img = data['img_restored']
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    global model
    if model is None:
        model = get_model(device)

    # img = aligned_img[..., ::-1].transpose(2, 0, 1) # RGB, CHW, [0, 255]
    img = cv2.resize(img, (256, 256))
    img_tensor = img2tensor(img / 255., bgr2rgb=True, float32=True)
    normalize(img_tensor, (.5, .5, .5), (.5, .5, .5), inplace=True)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        p_map = model(img_tensor)
        sum = torch.sum(p_map) / (256*256)
        ifqa_score = round(torch.mean(sum).detach().cpu().item(), 4)

    return ifqa_score