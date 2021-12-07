import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
    
    def forward(self, x):
        pass