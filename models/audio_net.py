import torch,torchvision
import torch.nn as nn
import torch.nn.functional as F


class Unet(nn.Module):
    def __init__(self,  use_dropout=False): 
        super(Unet, self).__init__()
        # construct unet structure

        K = 16 #  number of channels
        num_ds = 7 # number of downsampling layers
        nerve_grow_factor = 16 # nerve grow factor
        # this is the innermost layer of our network
        unet = UnetBlock(
            nerve_grow_factor * 8, nerve_grow_factor * 8, input_nc=None, subblock=None, innermost=True)
        # we have two layers that uses the dropout to prevent co-adaptation of neurons.    
        for i in range(num_ds - 5):
            unet = UnetBlock(
                nerve_grow_factor * 8, nerve_grow_factor * 8, 
                input_nc=None, subblock=unet, use_dropout=use_dropout)
        # 
        unet = UnetBlock(
            nerve_grow_factor * 4, nerve_grow_factor * 8, input_nc=None,
            subblock=unet)
        unet = UnetBlock(
            nerve_grow_factor * 2, nerve_grow_factor * 4, input_nc=None,
            subblock=unet)
        unet = UnetBlock(
            nerve_grow_factor, nerve_grow_factor * 2, input_nc=None,
            subblock=unet)
        unet = UnetBlock(
            K, nerve_grow_factor, input_nc=1,
            subblock=unet, outermost=True)
        self.batchNorm = nn.num_features(1)
        self.unet = unet

    def forward(self, x): 
        x = self.batchNorm(x)
        x = self.unet(x)
        return x


class UnetBlock(nn.Module):
    def __init__(self, outer_nc, inner_input_nc, input_nc=None,
                 subblock=None, outermost=False, innermost=False,
                 use_dropout=False, inner_output_nc=None, noSkipConnection=False):

        super(UnetBlock, self).__init__()
        self.outermost = outermost
        self.noSkipConnection = noSkipConnection
        # not innermost
        if input_nc is None:
            input_nc = outer_nc
        # innermost
        if innermost:
            inner_output_nc = inner_input_nc
        # the layers in between 
        elif inner_output_nc is None:
            inner_output_nc = 2 * inner_input_nc


        upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        # downrelu = nn.LeakyReLU(0.2, True)
        downrelu = nn.ReLU(True)
        uprelu = nn.ReLU(True)
        
        downnorm = nn.BatchNorm2d(inner_input_nc)
        upnorm = nn.BatchNorm2d(outer_nc)
        
        if innermost:
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=False)
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3,
                padding=1, bias=False)

            down = [downrelu, downconv]
            up = [uprelu, upsample, upconv, upnorm]
            model = down + up


        elif outermost:
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=False)
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3, padding=1)

            down = [downconv]
            up = [uprelu, upsample, upconv]
            model = down + [subblock] + up


        else:
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=False)
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3,
                padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upsample, upconv, upnorm]

            if use_dropout:
                model = down + [subblock] + up + [nn.Dropout(0.5)]
            else:
                model = down + [subblock] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost or self.noSkipConnection:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)#

if __name__ == "__main__":

    ## 
    net = Unet( use_dropout=True)
    # input image
    x    = torch.randn(1,1, 256, 256)
    net(x)
    print(net(x).shape)
