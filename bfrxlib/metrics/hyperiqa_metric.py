import torch
import torchvision
from PIL import Image
import numpy as np

from .hyperiqa.hyperiqa_net import HyperIQA
from facexlib.utils import load_file_from_url
from bfrxlib.utils.registry import METRIC_REGISTRY

MODEL_URL = 'https://github.com/xinntao/facexlib/releases/download/v0.2.0/assessment_hyperIQA.pth'
    
model = None
@METRIC_REGISTRY.register()
def calculate_hyperiqa(data, device=None, crop_border=50, **kargs):
    # Modified from hyperIQA/demo.py & facexlib/inference_hyperiqa.py
    img = data['img_restored']
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    global model
    if model is None:
        # assess_net = init_assessment_model(args.assess_model_name, half=False)
        model = HyperIQA(16, 112, 224, 112, 56, 28, 14, 7).to(device)
        model = model.eval()
        # load our pre-trained model on the koniq-10k dataset
        # model_hyper.load_state_dict((torch.load('./pretrained/koniq_pretrained.pkl')))
        hypernet_model_path = load_file_from_url(
            url=MODEL_URL, model_dir='bfrxlib/weights/assessment', progress=True, file_name=None)
        model.hypernet.load_state_dict((torch.load(hypernet_model_path, map_location=lambda storage, loc: storage)))

    # specified face transformation in original hyperIQA
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((512, 384)),
        torchvision.transforms.RandomCrop(size=224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # random crop 10 patches and calculate mean quality score
    pred_scores = []
    for i in range(10):
        if img.ndim == 2:
            img_tmp = img_tmp
        # BRG -> RGB
        img_tmp = img[..., ::-1][crop_border:, :-crop_border-1, ...] # RGB, HWC, [0, 255]
        # print(img_tmp.shape, img_tmp, img.ndim)
        
        img_tmp = Image.fromarray(img_tmp)
        
        img_tmp = transforms(img_tmp)
        img_tensor = torch.tensor(img_tmp.to(device)).unsqueeze(0)
        pred = model(img_tensor)  # 'paras' contains the network weights conveyed to target network

        # Quality prediction
        pred_scores.append(float(pred.item()))
    hyperiqa_score = np.mean(pred_scores)
    # quality score ranges from 0-100, a higher score indicates a better quality
    # print('Predicted quality score: %.2f' % score)

    return hyperiqa_score
