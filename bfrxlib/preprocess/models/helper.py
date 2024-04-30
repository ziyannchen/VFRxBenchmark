import torch

from facexlib.detection import init_detection_model


# TODO: Image/Video helper
class FaceSImage(object):
    def __init__(self, frame, det_model='retinaface_resnet50') -> None:
        self.det_net = init_detection_model(det_model, half=False)
        self.frame = frame

    def detect(self, face_score_thresh=0.97):
        with torch.no_grad():
            bboxes = self.det_net.detect_faces(self.frame, face_score_thresh)