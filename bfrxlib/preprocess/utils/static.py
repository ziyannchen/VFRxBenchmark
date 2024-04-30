import numpy as np
import os

DBmap = {'1': 'ytf', '2': 'ytceleb', '3': 'ytw'}


def read_pos_info(clip_name, pos_dir):
    # db, video_id, clip_id = clip_name.split('_')[:3]
    # base = '_'.join([db, video_id, clip_id])
    base = clip_name
    pos_file = os.path.join(pos_dir, base+'.txt')
    pos_info = np.loadtxt(pos_file)
    return pos_info


def get_landmarks2D(img_name, pos_info):
    frame_id = int(img_name.split('.')[0].split('_')[-1])
    # only store one face in a single clip
    pos_info = (pos_info[frame_id]).astype(np.float32) # pos info without confidence with length of 4 + 10
    bbox = pos_info[:5]
    landmark = np.array([[pos_info[i], pos_info[i + 1]] for i in range(4, 14, 2)])
    # print(landmark.shape)
    return bbox, landmark