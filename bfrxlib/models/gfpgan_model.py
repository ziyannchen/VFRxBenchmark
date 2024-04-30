import torch
from os import path as osp

from basicsr.utils import tensor2img
from basicsr.archs import build_network
from basicsr.utils.registry import MODEL_REGISTRY

from .base_model import BaseModel

@MODEL_REGISTRY.register()
class GFPGANModel(BaseModel):
    def __init__(self, opt):
        super(GFPGANModel, self).__init__(opt, 'gfpgan')

    def _inference(self, cropped_face_t):
        # FIXME: param, weight=weight
        output, _ = self.net(cropped_face_t, return_rgb=False)
        return output
