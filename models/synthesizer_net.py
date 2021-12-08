import torch
import torch.nn as nn
import torch.nn.functional as F

class SynthesizerNet(nn.Module):
    def __init__(self, fc_dim):
        super(SynthesizerNet, self).__init__()
        self.scale = nn.Parameter(torch.ones(fc_dim))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, image_features, sound_features):
        # Inputs:
        # image_features: (batch_size, num_channels)
        # sound_features: (batch_size, num_channels, HS, WS)
        
        # Output: 
        # z: (batch_size, HS, WS), where (HS,WS) represent height and width of each T-F representation
        
        # Note that: num_channels = K (in paper) = fc_dim (in code)
        
        B, C, HS, WS = sound_features.size
        image_features = image_features.view(B, 1, C)
        
        # forward pass
        z = torch.bmm(image_features * self.scale, sound_features.view(B, C, -1))
        z = z.view(B, 1, HS, WS)
        z = z + self.bias
        
        return z