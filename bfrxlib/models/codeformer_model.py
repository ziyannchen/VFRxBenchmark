import torch
from os import path as osp

from basicsr.utils import tensor2img
from basicsr.utils.registry import MODEL_REGISTRY

from .base_model import BaseModel


@MODEL_REGISTRY.register()
class CodeFormerModel(BaseModel):
    def __init__(self, opt):
        super(CodeFormerModel, self).__init__(opt, 'codeformer')

    def _inference(self, cropped_face_t):
        output = self.net(cropped_face_t, w=self.opt['fidelity_weight'], adain=True)
        return output[0]
