import torch
import numpy as np
from os import path as osp
from torchvision.transforms.functional import normalize

from basicsr.utils import tensor2img
from basicsr.utils.registry import MODEL_REGISTRY

from .base_model import BaseModel

@MODEL_REGISTRY.register()
class GCFSRModel(BaseModel):
    def __init__(self, opt):
        super(GCFSRModel, self).__init__(opt, 'gcfsr')

    def _load_model(self):
        assert 'pretrain_model_url' in self.opt['path'] or 'pretrain_model' in self.opt['path'], 'Pretrained model not given!'
        if 'pretrain_model_url' in self.opt['path'] and self.opt['path']['pretrain_model_url'] is not None:
            ckpt_path = load_file_from_url(url=self.opt['path']['pretrain_model_url'], 
                                            model_dir=self.opt['path']['weights_root'], progress=True, file_name=None)
        else:
            ckpt_path = osp.join(self.opt['path']['weights_root'], self.opt['path']['pretrain_model'])
        checkpoint = torch.load(ckpt_path)
        self.net.load_state_dict(checkpoint, strict=True)
        self.net.eval()
    
    def _inference(self, cropped_face_t):
        output, _ = self.net(cropped_face_t)
        return output
