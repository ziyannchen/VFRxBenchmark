import torch
from os import path as osp

from basicsr.utils import tensor2img
from basicsr.utils.registry import MODEL_REGISTRY

from .base_model import FOSBaseVideoModel


@MODEL_REGISTRY.register()
class FOSEDVRModel(FOSBaseVideoModel):
    def __init__(self, opt):
        super(FOSEDVRModel, self).__init__(opt, 'edvr')
