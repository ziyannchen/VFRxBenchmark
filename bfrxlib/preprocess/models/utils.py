import numpy as np

def clip_bbox(bbox: np.array, w: int, h: int) -> np.array:
    """Clip the bbox whthin the boundary.

    Args:
        bbox (np.array): [x1, y1, x2, y2]
        w (int): frame width. Max index of x should be w-1.
        h (int): frame height. Max index of y shoule be h-1.

    Returns:
        np.array: clipped bbox
    """
    bbox[..., [0, 2]] = np.clip(bbox[..., [0, 2]], a_min=0, a_max=w-1)
    bbox[..., [1, 3]] = np.clip(bbox[..., [1, 3]], a_min=0, a_max=h-1)
    return bbox

def add_bbox_margin(bbox: np.array, bbox_margin: int, frame_w: int, frame_h: int) -> np.array:
    """add margin to the bbox.

    Args:
        bbox (np.array): _description_
        bbox_margin (int): _description_
        frame_w (int): _description_
        frame_h (int): _description_

    Returns:
        np.array: bbox with margin added.
    """
    # bbox[..., 0] = np.maximum(bbox[..., 0] - bbox_margin, 0)
    # bbox[..., 1] = np.maximum(bbox[..., 1] - bbox_margin, 0)
    # bbox[..., 2] = np.minimum(bbox[..., 2] + bbox_margin, frame_w)
    # bbox[..., 3] = np.minimum(bbox[..., 3] + bbox_margin, frame_h)
    bbox[..., [0, 1]] = bbox[..., [0, 1]] - bbox_margin
    bbox[..., [2, 3]] = bbox[..., [2, 3]] + bbox_margin
    bbox = clip_bbox(bbox, frame_w, frame_h)
    return bbox