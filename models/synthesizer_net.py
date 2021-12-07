import torch
import torch.nn as nn
import torch.nn.functional as F

class SynthesizerNet(nn.Module):
    def __init__(self, fc_dim):
        super(SynthesizerNet, self).__init__()
        self.scale = nn.Parameter(torch.ones(fc_dim))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, img_features, sound_features):
        # Inputs:
        # img_features: (batch_size, num_img_channels)
        # sound_features: (batch_size, num_audio_channels, HS, WS)
        
        # K = num_img_channels = num_audio_channels = fc_dim
        
        B, C, HS, WS = sound_features.size
        img_features = img_features.view(B, 1, C)
        
        # forward pass
        z = torch.bmm(img_features * self.scale, sound_features.view(B, C, -1))
        z = z.view(B, 1, HS, WS)
        z = z + self.bias
        
        return pred_mask
