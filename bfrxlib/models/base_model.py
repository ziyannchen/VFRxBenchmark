import torch
import os
from collections import OrderedDict
from torchvision.transforms.functional import normalize

from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.base_model import BaseModel as BasicBaseModel

from bfrxlib.archs import build_network
from bfrxlib.utils import yaml_load, yaml_find

class BaseModel(BasicBaseModel):
    '''
    The base model originally for inference, providing api for pre- and post-processing.
    Any model with a different pre- or post-processing pipeline should inherit this class and rewrite the methods.
    '''
    def __init__(self, opt, name='base'):
        # Benchmarking SOTA methods based on standard data (FOS default: aligned, lq size=512)
        self.name = name
        self.opt = opt
        self.inference_type = 'single_image' # SingleImage face restoration inference pipeline by default
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')

        self.net = build_network(self.opt['network_g']).to(self.device)
        self._load_model()

    def _load_model(self):
        assert 'pretrained_model_url' in self.opt['path'] or 'pretrained_model' in self.opt['path'], 'Pretrained model not given!'

        PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if self.opt['path']['pretrained_model_url'] is not None:
            ckpt_path = load_file_from_url(url=self.opt['path']['pretrained_model_url'], 
                                            model_dir=os.path.join(PROJ_DIR, 'weights'), progress=True, file_name=None)
        else:
            ckpt_path = self.opt['path']['pretrained_model']
        print(ckpt_path)
        checkpoint = torch.load(ckpt_path)

        keyname = 'params'
        if 'params_ema' in checkpoint:
            keyname = 'params_ema'
        self.net.load_state_dict(checkpoint[keyname], strict=True)
        self.net.eval()

    # the preprocessing has been encapsulated into the dataloader
    def _preprocess(self, cropped_face):
        cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True).to(self.device)
        normalize(cropped_face_t, self.opt['dataset']['mean'], self.opt['dataset']['std'], inplace=True)
        cropped_face_t = cropped_face_t

        return cropped_face_t
    
    def _inference(self, cropped_face_t):
        return self.net(cropped_face_t)
    
    def _postprocess(self, output):
        return tensor2img(output, rgb2bgr=True, min_max=(-1, 1))

    def __call__(self, lq):
        try:
            output = self._inference(lq.to(self.device))
        except Exception as error:
            print(f'\tError! Failed inference for '+self.name+f': {error}')
            output = lq

        restored_face = self._postprocess(output)
        return restored_face


@MODEL_REGISTRY.register()
class FOSBaseVideoModel(BaseModel):
    def __init__(self, opt, name='base_video_model'):
        super(FOSBaseVideoModel, self).__init__(opt, name)
        self.inference_type = 'video' # Multi-frame face restoration inference pipeline

    def _preprocess(self, cropped_face):
        cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True).to(self.device)
        cropped_face_t = cropped_face_t

        return cropped_face_t

    def _postprocess(self, outputs):
        '''
        returns numpy array of restored frames
        Args:
            outputs: The outputs of a video model. 
                torch.tensor with shape like [batch_size, seq_len, C, H, W] or [batch_size, C, H, W]

        Returns:
            restored_imgs: A mini-batch length list of numpy images with shape like [H, W, C]
        '''
        # postprocessing of the video/clip seq
        restored_imgs = []

        if outputs.dim() == 4:
            # the seq_len channel is squeezed in the forward propagation of the video model
            outputs = outputs.unsqueeze(0)
        outputs = list(outputs)

        # loop over the mini-batch
        for output in outputs:
            output = tensor2img(output, min_max=(0, 1))
            restored_imgs.append(output)

        return restored_imgs