import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from functools import partial


class ResnetDilate(nn.Module):

    def __init__(self, K=16):
        super(ResnetDilate, self).__init__()

        original_net = torchvision.models.resnet18(pretrained=True)

        # Remove the stride of the last residual block
        original_net.layer4.apply(partial(self._nostride_dilate, dilate=2))

        # Remove the last two layers of the ResNet - average pooling & fc
        self.features = nn.Sequential(*list(original_net.children())[:-2])

        # Add 3x3 convolution layer with K output channels
        self.fc = nn.Conv2d(512, K, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.size(0), x.size(1))

        return x

    def forward_multiframe(self, X):
        (B, C, T, H, W) = X.size()
        X = X.permute(0, 2, 1, 3, 4).contiguous()
        X = X.view(B*T, C, H, W)

        X = self.features(X)
        X = self.fc(X)

        (_, C, H, W) = X.size()
        X = X.view(B, T, C, H, W)
        X = X.permute(0, 2, 1, 3, 4)
        X = F.adaptive_max_pool3d(X, 1)
        X = X.view(B, C)

        return X

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)
