from os import path as osp
import numpy as np
from collections import OrderedDict
from torch.utils import data as data
from torchvision import transforms
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paths_from_lmdb
from basicsr.utils import FileClient, imfrombytes, img2tensor, rgb2ycbcr, scandir
from basicsr.utils.registry import DATASET_REGISTRY

from bfrxlib.utils import is_gray, yaml_find


@DATASET_REGISTRY.register()
class FosSingleImageDataset(data.Dataset):
    """ Modified from BasicSR SingleImageDataset.

    Read only lq images in the test phase.
    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc).

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path list file of path information. Scan folder as default if None.
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt, transform_fn=None):
        super(FosSingleImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend'] if hasattr(opt, 'io_backend') else OrderedDict(type='disk')

        self.lq_data_info = {'key': 'lq', 'folder': opt['dataroot_lq'], 'paths': []}
        self.lq_data_info = self._scandir(self.lq_data_info)

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_data_info['folder']]
            self.io_backend_opt['client_keys'] = [self.lq_data_info['key']]

        self.transform_fn = transform_fn if transform_fn is not None else self.img2tensor_norm

    def _scandir(self, data_info, full_path=False):
        folder = data_info['folder']
        if self.io_backend_opt['type'] == 'lmdb':
            data_info['paths'] = paths_from_lmdb(folder)
        elif yaml_find(self.opt, 'meta_info_file') is not None:
            with open(self.opt['meta_info_file'], 'r') as fin:
                data_info['paths'] = [line.rstrip().split(' ')[0] for line in fin]
                if full_path:
                    data_info['paths'] = [osp.join(folder, v) for v in data_info['paths']]
        else:
            data_info['paths'] = sorted(list(scandir(folder, full_path=full_path)))

        return data_info

    def img2tensor_norm(self, img):
        '''
        img: image with bgr channels
        '''
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_tensor = img2tensor(img / 255., bgr2rgb=True, float32=True)
        if img.shape[-1] != self.opt['face_size']:
            img_tensor = transforms.Resize((self.opt['face_size'], self.opt['face_size']))(img_tensor)

        # normalize
        if self.opt['mean'] is not None or self.opt['std'] is not None:
            normalize(img_tensor, self.opt['mean'], self.opt['std'], inplace=True)
        return img_tensor

    def _read_process(self, lq_path):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        # load lq image
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        is_grayscale = is_gray(img_lq*255, threshold=5)

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        img_lq_tensor = self.transform_fn(img_lq * 255)

        return img_lq_tensor, is_grayscale

    def __getitem__(self, index):

        lq_basename = self.lq_data_info['paths'][index]
        lq_path = osp.join(self.lq_data_info['folder'], lq_basename)
        img_lq_tensor, is_grayscale = self._read_process(lq_path)

        return {
            'lq': img_lq_tensor, 
            'lq_path': lq_path, 
            'lq_basename': osp.splitext(lq_basename)[0], 
            'is_gray': is_grayscale
        }

    def __len__(self):
        return len(self.lq_data_info['paths'])


@DATASET_REGISTRY.register()
class FosPairedImageDataset(FosSingleImageDataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """
    def __init__(self, opt, transform_fn=None):
        super(FosPairedImageDataset, self).__init__(opt, transform_fn)
        assert 'dataroot_gt' in self.opt, 'The ground-truth data directory needs to be provided!'
        self.gt_data_info = {'key': 'gt', 'folder': opt['dataroot_gt'], 'paths': []}
        self.gt_data_info = self._scandir(self.gt_data_info, full_path=True)

    def __getitem__(self, index):
        lq_basename = self.lq_data_info['paths'][index]
        lq_path = osp.join(self.lq_data_info['folder'], lq_basename)
        gt_path = self.gt_data_info['paths'][index]

        img_lq_tensor, _ = self._read_process(lq_path)
        img_gt_tensor, is_grayscale = self._read_process(gt_path)

        return {'lq': img_lq_tensor, 
                'gt': img_gt_tensor, 
                'lq_path': lq_path, 
                'gt_path': gt_path, 
                'lq_basename': osp.splitext(lq_basename)[0], # gt share the same basename with lq
                'is_gray': is_grayscale}

    