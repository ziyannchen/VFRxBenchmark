from basicsr.utils.registry import LOSS_REGISTRY

import torch
from torch import nn


@LOSS_REGISTRY.register()
class OhemCELoss(nn.Module):
    def __init__(self, thresh, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.ignore_lb = ignore_lb
        self.n_min = None
        self.thresh = thresh
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        # print('loss', loss.shape)
        loss, _ = torch.sort(loss, descending=True)

        # filter loss values to help balance the foreground and background classes
        if self.n_min is None:
            self.n_min = logits.shape[0] * logits.shape[-2] * logits.shape[-1]//16
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)