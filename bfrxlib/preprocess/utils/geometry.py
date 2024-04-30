import numpy as np

def enlarge_side(enlarge_size: float, x0, x1, min_x, max_x):
    '''To legally enlarge the side. It can be a side shrinkage when enlarge_size < 0.'''
    enlarge_size /= 2
    x1 += np.floor(enlarge_size)
    x0 -= np.ceil(enlarge_size)
    # print('enlarge_side: ', x1, x0, max_x, min_x)
    if x0 < min_x:
        x1 += min_x - x0
        x0 = min_x
    elif x1 > max_x:
        x0 -= x1 - max_x
        x1 = max_x
    return x0, x1
    
    
def check_bcomp(x0, x1, y0, y1, min_x, max_x, min_y, max_y):
    '''when enlarge the x side:
    yi_bomp: yi boiundary compensation, in case the x side is too short to have enough space to be enlarged
    then the length of the y side has to be reduced (e.g. when max_x < h)
    '''
    if x0 >= min_x and x1 <= max_x and y0 >= min_y and y1 <= max_y:
        return x0, x1, y0, y1

    y0_bcomp = min(x0-min_x, 0)     # a negtive num, to reduce the y side
    y1_bcomp = min(max_x-x1, 0)     # a negtive num
    y_bcomp = (y0_bcomp + y1_bcomp)

    y0, y1 = enlarge_side(y_bcomp, y0, y1, min_y, max_y)
    x0 = max(x0, min_x)
    x1 = min(x1, max_x)
    return x0, x1, y0, y1


def enlarge_bbox_square(bbox, max_x, max_y, min_y=0, min_x=0):
    '''To reshape an arbitrary-shaped bbox to a square-shape'''
    x0, y0, x1, y1 = bbox
    # print(x0, x1, y0, y1, max_x, max_y)
    x0, y0 = max(x0, min_x), max(y0, min_y)
    x1, y1 = min(x1, max_x), min(y1, max_y)
    # print(f'x0={x0}, x1={x1}, y0={y0}, y1={y1}')
    w, h = x1 - x0, y1 - y0
    assert w > 0 and w <= max_x and h > 0 and h <= max_y, 'Error, x1 < x0 or y1 < y0!'

    enlarge_size = abs(h - w)
    if w < h:
        x0, x1 = enlarge_side(enlarge_size, x0, x1, min_x, max_x)
        # print('enlarge bbox : w<h', x0, x1)
        # print(f'w={w} < h={h}: x side enlarged. x0={x0}, x1={x1}, y0={y0}, y1={y1}')
        # to reduce y side in case x side is too short (when x0 or x1 is negtive)
        x0, x1, y0, y1 = check_bcomp(x0, x1, y0, y1, min_x, max_x, min_y, max_y)
    elif w > h:
        y0, y1 = enlarge_side(enlarge_size, y0, y1, min_y, max_y)
        # print('enlarge bbox : w>h', y0, y1)
        # print(f'w={w} > h={h}: y side enlarged. x0={x0}, x1={x1}, y0={y0}, y1={y1}')
        y0, y1, x0, x1 = check_bcomp(y0, y1, x0, x1, min_y, max_y, min_x, max_x)
    # print('after chek comp: ', x0, y0, x1, y1)
    # print(x0, x1, y0, y1)
    return np.array([x0, y0, x1, y1], dtype=int)


def iou(bboxes, bbox_gt):
    test_x0 = bboxes[:, 0]
    test_y0 = bboxes[:, 1]
    test_x1 = bboxes[:, 2]
    test_y1 = bboxes[:, 3]
    # get intersection
    xx1 = np.maximum(test_x0, bbox_gt[0])
    yy1 = np.maximum(test_y0, bbox_gt[1])
    xx2 = np.minimum(test_x1, bbox_gt[2])
    yy2 = np.minimum(test_y1, bbox_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    itersection = w * h
    union = (test_x1 - test_x0) * (test_y1 -test_y0) + (bbox_gt[2] - bbox_gt[0]) * (bbox_gt[3] - bbox_gt[1]) - itersection
    return itersection / union