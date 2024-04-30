import numpy as np
import torch
import cv2
from torchvision.transforms import functional as F

from facexlib.detection import init_detection_model
from facexlib.headpose import init_headpose_model
from facexlib.utils.misc import img2tensor

class HeadposeEst():
    def __init__(self, net_name='hopenet', device='cuda'):
        self.device = torch.device(device)
        self.net = init_headpose_model(net_name, device=device)

    def __call__(self, face_img, bbox=None):
        with torch.no_grad():
            if bbox is None:
                det_net = init_detection_model('retinaface_resnet50', half=False)
                bboxes = det_net.detect_faces(face_img, 0.97)
                bbox = list(map(int, bboxes[0]))

            thld = 10
            h, w = face_img.shape[:2]
            top = max(bbox[1] - thld, 0)
            bottom = min(bbox[3] + thld, h)
            left = max(bbox[0] - thld, 0)
            right = min(bbox[2] + thld, w)

            det_face = face_img[top:bottom, left:right, :].astype(np.float32) / 255.

            # resize
            det_face = cv2.resize(det_face, (224, 224), interpolation=cv2.INTER_LINEAR)
            det_face = img2tensor(np.copy(det_face), bgr2rgb=False)

            # normalize
            F.normalize(det_face, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)
            det_face = det_face.unsqueeze(0).to(self.device)

            yaw, pitch, roll = self.net(det_face)
            # visualize_headpose(face_img, yaw, pitch, roll)
            
        return yaw, pitch, roll
