import cv2
import numpy as np

def vis_parsing_maps(img, parsing_anno, stride, save_anno_path=None, save_vis_path=None):
    part_colors = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], 
                [51, 51, 255], [204, 0, 204], [0, 255, 255],
                [255, 204, 204], [102, 51, 0], [0, 204, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], 
                [0, 0, 204], # 13: 'hair' red
                [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51]]
    assert np.max(parsing_anno) <= len(part_colors), f'Maximum annotation {np.max(parsing_anno)} exceeds predefined {len(part_colors)} class(es).'
    vis_parsing_anno = parsing_anno.copy().round().astype(np.uint8)
    if vis_parsing_anno.ndim == 2:
        # to match dim shape with img
        vis_parsing_anno = np.expand_dims(vis_parsing_anno, axis=2)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    if save_anno_path is not None:
        cv2.imwrite(save_anno_path, vis_parsing_anno)

    if save_vis_path is not None:
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
        num_of_class = np.max(vis_parsing_anno)
        for pi in range(1, num_of_class + 1):
            index = np.where(vis_parsing_anno == pi)
            vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        vis_im = cv2.addWeighted(img.astype(np.uint8), 0.4, vis_parsing_anno_color, 0.6, 0)

        cv2.imwrite(save_vis_path, vis_im)
    return vis_im