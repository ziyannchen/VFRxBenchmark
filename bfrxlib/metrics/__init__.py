import importlib
from copy import deepcopy
from os import path as osp

from basicsr.utils import get_root_logger, scandir
from bfrxlib.utils.registry import METRIC_REGISTRY

__all__ = ['calculate_metric']
metric_folder = osp.dirname(osp.abspath(__file__))
metric_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(metric_folder) if v.endswith('_metric.py')]
# import all the metric modules
_metric_modules = [importlib.import_module(f'bfrxlib.metrics.{file_name}') for file_name in metric_filenames]


def init_metric_model(opt):
    # TODO: more standard coding. provide a unified api to define a model in entry script
    pass

def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
