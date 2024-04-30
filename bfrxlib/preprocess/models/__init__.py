from .tracker import *
from .detector import *
from .headpose import *
from .descriptor import *
from .utils import clip_bbox

import torch
from facexlib.utils import load_file_from_url

def init_recognition_model(model_name, device='cuda', model_rootpath=None):
    ''' An extention of init_recognition_model in facexlib. 
        Currently only support arcface with different backbones.
        Ref: 1. facexlib (https://github.com/xinntao/facexlib/blob/master/facexlib/recognition/__init__.py).
             2. insightface (https://github.com/deepinsight/insightface).
    Args:
        model_name (str): The name of the face recognition model. 
            Options: ['arcface_ir_se50', 'arcface_resnet18'].
    '''
    if model_name == 'arcface_ir_se50':
        from facexlib.recognition.arcface_arch import Backbone
        model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se').to('cuda').eval()
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/recognition_arcface_ir_se50.pth'
    elif 'arcface_resnet18' in model_name:
        from .arcface.resnet import ResNet18_Face
        model = ResNet18_Face(use_se=False)
        # TODO: upload the model to github to download automatically.
        # FIXME: this is an invalid url
        model_url = 'https://github.com/releases/download/arcface_resnet18_110.pth'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(
        url=model_url, model_dir='facexlib/weights', progress=True, file_name=None, save_dir=model_rootpath)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)
    return model