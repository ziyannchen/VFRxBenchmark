import torch
from os import path as osp
import numpy as np
import cv2

from basicsr.utils.registry import MODEL_REGISTRY

from .base_model import BaseModel

@MODEL_REGISTRY.register()
class GPENModel(BaseModel):
    def __init__(self, opt):
        super(GPENModel, self).__init__(opt, 'gpen')
    
    def _load_model(self):
        checkpoint = torch.load(osp.join(self.opt['path']['weights_root'], self.opt['path']['pretrain_network']))
        self.net.load_state_dict(checkpoint, strict=True)
        self.net.eval()

    def _inference(self, cropped_face_t):
        out, _ = self.net(cropped_face_t)
        return out
