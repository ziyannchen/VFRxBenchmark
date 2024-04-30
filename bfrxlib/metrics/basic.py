import numpy as np
import torch
import torch.nn.functional as F

def large_face_motion(bbox_matrix, crop_bbox, thresh: float = 0.5):
    '''
    To judge if the face motion is too large.
    If the face bbox width is far smaller than the crop bbox, than it is viewed as a large motin clip.
    face_motion_dist_rate = crop_bbox_size / face_bbox_size.
    Note that the fade bbox is the ones with margin.
    '''
    # approximate the face width
    face_size = np.sqrt((bbox_matrix[:, 2] - bbox_matrix[:, 0])**2 + (bbox_matrix[:, 3] - bbox_matrix[:, 1])**2)

    # compute face motion
    frame_size = np.sqrt((crop_bbox[2] - crop_bbox[0])**2 + (crop_bbox[3] - crop_bbox[1])**2)

    motion_rate = face_size/frame_size
    # print(motion_rate)
    if any(motion_rate < thresh):
        return True
    return False

def cal_distance(feat_matrix_1, feat_matrix_2, dis_type: str = 'l2'):
    '''
    Calculate distance of vectors.

    Args:
        type: str. options: ['l2', 'cosine'].
    '''
    assert type(feat_matrix_1) == type(feat_matrix_2), \
        f'Inputs should be the same type, but got {type(feat_matrix_1)} and {type(feat_matrix_2)}.'

    if isinstance(feat_matrix_1, np.ndarray):
        feat_matrix_1 = torch.from_numpy(feat_matrix_1)
        feat_matrix_2 = torch.from_numpy(feat_matrix_2)
    
    sim_dict = {
        'l2': lambda x1, x2: torch.linalg.norm((x1 - x2), ord=2, dim=-1, keepdim=False),
        'cosine': lambda x1, x2: 1 - F.cosine_similarity(x1, x2, dim=-1)
        }
    assert dis_type in sim_dict.keys(), \
        f'Similarity measure metric only support type in {list(sim_dict.keys())}, but got {dis_type}.'

    return sim_dict[dis_type](feat_matrix_1, feat_matrix_2)