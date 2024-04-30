import torchvision
import torch
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

from facexlib.detection import init_detection_model
from facexlib.assessment import init_assessment_model

def hyperIQA(face_img: np.array, bbox: list=None):
    """Face Image Quality Assessment based on hyperIQA.

    Args:
        face_img (np.array): 
        bbox (list, optional): face bbox. Defaults to None.

    Returns:
        float: face quality score. [0, 100]
    """

    with torch.no_grad():
        if bbox is None:
            det_net = init_detection_model('retinaface_resnet50', half=False)
            bboxes = det_net.detect_faces(face_img, 0.97)
            bbox = list(map(int, bboxes[0]))

        # specified face transformation in original hyperIQA
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((512, 384)),
            torchvision.transforms.RandomCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        assess_net = init_assessment_model('hypernet', half=False)
        
        pred_scores = []
        # BRG -> RGB
        img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        for i in tqdm(range(10), desc='Assessing the quality of the detected face...'):
            detect_face = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            detect_face = Image.fromarray(detect_face)

            detect_face = transforms(detect_face)
            detect_face = torch.tensor(detect_face.cuda()).unsqueeze(0)

            pred = assess_net(detect_face)
            pred_scores.append(float(pred.item()))
        score = np.mean(pred_scores)

        return score