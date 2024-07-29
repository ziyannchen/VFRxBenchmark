import numpy as np
import torch
import cv2
from tqdm import tqdm

from facexlib.tracking.sort import SORT
from facexlib.utils.face_restoration_helper import get_largest_face

from .utils import clip_bbox

def resize(image, max_size):
    height, width = image.shape[:2]
    scale = 1
    if max(height, width) > max_size:
        if height > width:
            scale = max_size / height
        else:
            scale = max_size / width

        # 计算新的尺寸
        new_height = int(height * scale)
        new_width = int(width * scale)

        # 调整图片大小
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return image, scale

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
    def __init__(self, det_net, *, input_size = 512, face_score_thresh=0.97, detect_interval=1, max_age=1, min_hits=2, iou_threshold=0.2):
        super(SortTracker, self).__init__(max_age, min_hits, iou_threshold)
        '''
        input_size: max input size. To save computing time, will resize the input to input_size if the exact size is larger.
        '''
        # face detector
        self.det_net = det_net
        self.face_score_thresh = face_score_thresh
        self.detect_interval = detect_interval

        self.input_size = input_size

    def _track_step(self, frame, only_keep_largest=False):
        frame, scale = resize(frame, self.input_size)
        img_size = frame.shape[:2]

        # detection face bboxes
        with torch.no_grad():
            bboxes = self.det_net.detect_faces(frame, self.face_score_thresh)
            if len(bboxes) and only_keep_largest:
                # only keep the largest face in every frame
                # returns largest bbox and the largest idx
                bboxes, _ = get_largest_face(bboxes, img_size[0], img_size[1])
                bboxes = bboxes[np.newaxis, ...] # [x0, y0, x1, y1, trace_face_id]

            bboxes[..., :4] = clip_bbox(bboxes[..., :4], img_size[1], img_size[0]) // scale
        
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