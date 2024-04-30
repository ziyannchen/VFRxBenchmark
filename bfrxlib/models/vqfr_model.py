import torch
from os import path as osp

from basicsr.utils.download_util import load_file_from_url
from basicsr.utils import tensor2img
from basicsr.utils.registry import MODEL_REGISTRY

from .base_model import BaseModel

@MODEL_REGISTRY.register()
class VQFRModel(BaseModel):
    def __init__(self, opt):
        super(VQFRModel, self).__init__(opt, 'vqfr')

    def _inference(self, cropped_face_t):
        output = self.net(cropped_face_t, fidelity_ratio=self.opt['fidelity_ratio'])['main_dec'][0]
        return output
