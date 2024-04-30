import os
import numpy as np
import cv2

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.metrics.niqe import niqe
from bfrxlib.utils.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
def calculate_niqe(data, crop_border=0, input_order='HWC', convert_to='y', **kwargs):
    """Modified from calculate_niqe of Basicsr.
    Calculate NIQE (Natural Image Quality Evaluator) metric.

    ``Paper: Making a "Completely Blind" Image Quality Analyzer``

    This implementation could produce almost the same results as the official
    MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip

    > MATLAB R2021a result for tests/data/baboon.png: 5.72957338 (5.7296)
    > Our re-implementation result for tests/data/baboon.png: 5.7295763 (5.7296)

    We use the official params estimated from the pristine dataset.
    We use the recommended block size (96, 96) without overlaps.

    Args:
        img (ndarray): Input image whose quality needs to be computed.
            The input image must be in range [0, 255] with float/int type.
            The input_order of image can be 'HW' or 'HWC' or 'CHW'. (BGR order)
            If the input order is 'HWC' or 'CHW', it will be converted to gray
            or Y (of YCbCr) image according to the ``convert_to`` argument.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        input_order (str): Whether the input order is 'HW', 'HWC' or 'CHW'.
            Default: 'HWC'.
        convert_to (str): Whether converted to 'y' (of MATLAB YCbCr) or 'gray'.
            Default: 'y'.

    Returns:
        float: NIQE result.
    """
    img = data['img_restored']
    METRICS_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    # we use the official params estimated from the pristine dataset.
    niqe_pris_params = np.load(os.path.join(METRICS_ROOT_DIR, 'niqe', 'niqe_pris_params.npz'))
    mu_pris_param = niqe_pris_params['mu_pris_param']
    cov_pris_param = niqe_pris_params['cov_pris_param']
    gaussian_window = niqe_pris_params['gaussian_window']

    img = img.astype(np.float32)
    if input_order != 'HW':
        img = reorder_image(img, input_order=input_order)
        if convert_to == 'y':
            img = to_y_channel(img)
        elif convert_to == 'gray':
            img = cv2.cvtColor(img / 255., cv2.COLOR_BGR2GRAY) * 255.
        img = np.squeeze(img)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border]

    # round is necessary for being consistent with MATLAB's result
    img = img.round()

    niqe_result = niqe(img, mu_pris_param, cov_pris_param, gaussian_window)

    return niqe_result