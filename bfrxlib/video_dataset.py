import os
from os import path as osp
import numpy as np
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paths_from_lmdb, generate_frame_indices
from basicsr.utils import FileClient, imfrombytes, img2tensor, rgb2ycbcr, scandir
from basicsr.utils.registry import DATASET_REGISTRY

from bfrxlib.utils import read_img_seq
from bfrxlib.data.img_dataset import FosSingleImageDataset


@DATASET_REGISTRY.register()
class FosVideoDataset(FosSingleImageDataset):
    """ Video frames dataset. Reference: BasicSR.
        Loading all frames from different clip into a single paths list in format like: 
        [subfolder1/0001.png, subfolder1/0002.png, ..., subfolder2/0001.png, ...]
        in which 'subfolder1/0001' is returned as lq_basename.
    
    Params:
        # all data infomation including image paths, folder name 'folder', and frame index of every frame
        data_info: {'key': '', 'paths': [], 'folder': [], 'idx': []}
    """
    def __init__(self, opt, transform_fn=None):
        super(FosVideoDataset, self).__init__(opt, transform_fn)
        self.opt['padding'] = 'reflection' if self.opt['seq_length'] == 1 else self.opt['padding']
        # lq_path: all lq img paths; folder: all clip frame folder path; idx: all index information of the lq imgs.

    def _scandir(self, data_info):
        root_folder = data_info['folder']
        data_info['folder'] = []
        data_info['idx'] = []

        folder_paths = [osp.join(root_folder, p) for p in sorted(os.listdir(root_folder))]
        paths_dict = {}
        for subfolder in folder_paths:
            subfolder_name = osp.basename(subfolder)
            img_paths = sorted(list(scandir(subfolder, full_path=True)))
            max_idx = len(img_paths)
            
            paths_dict[subfolder_name] = img_paths
            data_info['paths'].extend(img_paths)
            data_info['folder'].extend([subfolder_name] * max_idx)
            for i in range(max_idx):
                data_info['idx'].append(f'{i}/{max_idx}')

        data_info['paths_dict'] = paths_dict
        return data_info

    def __len__(self):
        return len(self.lq_data_info['paths'])

    # def transform_fn_seq(self, imgs):
    #     for index, img in enumerate(imgs):
    #         imgs[index] = self.transform_fn(img)
    #     return imgs

    def __getitem__(self, index):
        # load lq image clip seq
        folder = self.lq_data_info['folder'][index]
        # current index(center frame)
        idx, max_idx = self.lq_data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        # border = self.data_info['border'][index]
        lq_path = self.lq_data_info['paths'][index]
        paths_dict = self.lq_data_info['paths_dict']

        # Each fetch to generate a valid seq for inference centered by the current frame
        # select_idx e.g. [2, 1, 0, 1, 2] based on reflection padding
        select_idx = generate_frame_indices(idx, max_idx, self.opt['seq_length'], padding=self.opt['padding'])
        img_paths_lq = [paths_dict[folder][i] for i in select_idx]
        data_lq = read_img_seq(img_paths_lq, transform_fn=self.transform_fn, judge_gray=True) # (t[num_frames], c, h, w)
        imgs_lq_seq, is_grayscale = data_lq['imgs'], data_lq['is_gray']

        return {'lq': imgs_lq_seq,      # multi-frame seq tensor with t length in a shape of (t[num_frames], c, h, w)
                'lq_path': lq_path,      # the path of the center frame in the seq
                'is_gray': is_grayscale,
                'folder': folder,       # lq frames folder name
                'lq_basename': osp.splitext(osp.join(folder, osp.basename(lq_path)))[0],
                'frame_idx': self.lq_data_info['idx'][index],  # e.g., 0/99.
            }


@DATASET_REGISTRY.register()
class FosPairedVideoDataset(FosVideoDataset):
    def __init__(self, opt, transform_fn=None):
        super(FosPairedVideoDataset, self).__init__(opt, transform_fn)
        # lq_path: all lq img paths; folder: all clip frame folder path; idx: all index information of the lq imgs.
        self.gt_data_info = self._scandir({'key': 'gt', 'paths': [], 'folder': self.opt['dataroot_gt']})
        self.transform_fn = transform_fn

    def __getitem__(self, index):
        # load seq frames from both lq and gt seq dir
        folder = self.lq_data_info['folder'][index]

        # current index(center frame) (idx info of the lq and gt should be the same)
        idx, max_idx = self.lq_data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        # border = self.data_info['border'][index]
        lq_path = self.lq_data_info['paths'][index]
        gt_path = self.gt_data_info['paths'][index]

        lq_paths_dict = self.lq_data_info['paths_dict']
        gt_paths_dict = self.gt_data_info['paths_dict']

        # generate seq in format of frame indexs (idx info of the lq and gt should be the same)
        select_idx = generate_frame_indices(idx, max_idx, self.opt['seq_length'], padding=self.opt['padding'])

        img_paths_lq = [lq_paths_dict[folder][i] for i in select_idx]
        data_lq = read_img_seq(img_paths_lq, transform_fn=self.transform_fn, judge_gray=True) # (t[num_frames], c, h, w)
        imgs_lq_seq, is_grayscale = data_lq['imgs'], data_lq['is_gray']

        img_paths_gt = [gt_paths_dict[folder][i] for i in select_idx]
        data_gt = read_img_seq(img_paths_gt, transform_fn=self.transform_fn) # (t[num_frames], c, h, w)
        imgs_gt_seq = data_lq['imgs']

        return {'lq': imgs_lq_seq,      # multi-frame seq tensor with t length in a shape of (t[num_frames], c, h, w)
                'lq_path': lq_path,      # the path of the center frame in the seq
                'lq_basename': osp.splitext(osp.join(folder, osp.basename(lq_path))[0]), 
                'is_gray': is_grayscale,
                'gt': imgs_gt_seq,      # multi-frame seq tensor with t length in a shape of (t[num_frames], c, h, w)
                'gt_path': gt_path,
                'folder': folder,       # frames folder name
                'frame_idx': self.lq_data_info['idx'][index],  # e.g., 0/99
            }