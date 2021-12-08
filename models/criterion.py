import torch
import torch.nn as nn
import torch.nn.functional as F

class WL1Loss(nn.Module):
    def __init__(self):
        super(WL1Loss, self).__init__()

    def forward(self, pred, target, weight):
        return torch.mean(weight * torch.abs(pred - target))