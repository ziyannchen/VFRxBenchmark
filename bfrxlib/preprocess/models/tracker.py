import numpy as np
import torch
import cv2
from tqdm import tqdm

from facexlib.detection import init_detection_model
from facexlib.tracking.sort import SORT
from facexlib.utils.face_restoration_helper import get_largest_face

from .utils import clip_bbox

def interp(bbox1, bbox2, order='center', mode='linear'):
    '''
    order:
        'pre': x bbox1 bbox2
        'center': bbox1 x bbox2
        'post': bbox1 bbox2 x
    '''
    
    if mode == 'linear':
        if order == 'center':
            return bbox1 + (bbox2 - bbox1) / 2
        elif order == 'pre':
            return bbox1 - (bbox2 - bbox1)
        elif order == 'post':
            return bbox2 + (bbox2 - bbox1)
    elif mode == 'quadratic':
        return bbox1 + (bbox2 - bbox1) / 2
    else:
        raise NotImplementedError(f'interp mode {mode} not implemented')

class SortTracker(SORT):
    def __init__(self, det_net, face_score_thresh=0.97, detect_interval=1, max_age=1, min_hits=2, iou_threshold=0.2):
        super(SortTracker, self).__init__(max_age, min_hits, iou_threshold)
        # face detector
        self.det_net = det_net
        self.face_score_thresh = face_score_thresh
        self.detect_interval = detect_interval

    def _track_step(self, frame, only_keep_largest=False):
        img_size = frame.shape[:2]

        # detection face bboxes
        with torch.no_grad():
            bboxes = self.det_net.detect_faces(frame, self.face_score_thresh)
            if len(bboxes) and only_keep_largest:
                # only keep the largest face in every frame
                # returns largest bbox and the largest idx
                bboxes, _ = get_largest_face(bboxes, img_size[0], img_size[1])
                bboxes = bboxes[np.newaxis, ...]

            bboxes[..., :4] = clip_bbox(bboxes[..., :4], img_size[1], img_size[0])
        
        return bboxes

    def track(self, frame_list, only_keep_largest=False, frame_interpolate=False):
        pbar = tqdm(total=len(frame_list), unit='frames')
        tracker_results = []
        for frame_id, frame in enumerate(frame_list):
            bboxes = self._track_step(frame, only_keep_largest)

            tracker_results.append([None, bboxes])
            pbar.update(1)
            pbar.set_description(f'{frame_id}: detect {len(bboxes)} faces')

        for frame_id, (_, bboxes) in enumerate(tracker_results):
            if len(bboxes) == 0 and frame_interpolate:
                # TODO: to support frame interpolation
                pass
                print('Interpolating')
            face_list = bboxes[..., :5]
            additional_attr = bboxes[..., 4]
            trackers = self.update(np.array(face_list), frame_list[frame_id].shape[:2], additional_attr, self.detect_interval)
            tracker_results[frame_id][0] = trackers
        
        return tracker_results