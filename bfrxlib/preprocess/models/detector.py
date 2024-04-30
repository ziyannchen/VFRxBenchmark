import numpy as np

from facexlib.detection import init_detection_model
from facexlib.detection.retinaface import RetinaFace
from facexlib.detection.align_trans import warp_and_crop_face

class Detector(RetinaFace):
    def __init__(self, det_name='resnet50', half=False, phase='test'):
        super(Detector, self).__init__(det_name, half, phase)

    def align_from_landmark(self, face_image, landmark):
        facial5points = [[landmark[2 * j], landmark[2 * j + 1]] for j in range(5)]
        warped_face = warp_and_crop_face(np.array(face_image)[np.newaxis, ...], facial5points, self.reference, crop_size=(112, 112))
        return warped_face