import torch
from os import path as osp

from basicsr.utils.download_util import load_file_from_url
from basicsr.utils import tensor2img
# from bfrxlib.utils.registry import ARCH_REGISTRY
from basicsr.archs import build_network
from basicsr.utils.registry import MODEL_REGISTRY

from .base_model import BaseModel


@MODEL_REGISTRY.register()
class RestoreFormerModel(BaseModel):
    def __init__(self, opt):
        super(RestoreFormerModel, self).__init__(opt, 'restoreformer')
    
    def _load_model(self):
        ckpt_path = load_file_from_url(url=self.opt['path']['pretrain_model_url'], 
                                        model_dir=self.opt['path']['weights_root'], progress=True, file_name=None)
        # ckpt_path = osp.join(self.opt['path']['weights_root'], self.opt['path']['pretrain_network'])
        
        checkpoint = torch.load(ckpt_path)['state_dict']
        prefix = 'vqvae.'

        state_dict = self.net.state_dict()
        require_keys = state_dict.keys()
        keys = [k[len(prefix):] for k in checkpoint.keys()]
        # un_pretrained_keys = []
        for k in require_keys:
            if k not in keys: 
                # miss 'vqvae.'
                print(k, 'k in keys: ', k in keys)
                if k in keys:
                    state_dict[k] = checkpoint[prefix+k]
                # else:
                #    # un_pretrained_keys
                #     un_pretrained_keys.append(k)
            else:
                state_dict[k] = checkpoint[prefix+k]

        self.net.load_state_dict(state_dict, strict=True)
        self.net.eval()

    def _inference(self, cropped_face_t):
        output, _ = self.net(cropped_face_t)
        return output.squeeze(0)
