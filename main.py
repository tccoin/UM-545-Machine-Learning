import torch
import torchvision
import torch.nn.functional as F

from models.audio_net import Unet
from models.vision_net import ResnetDilated
from models.synthesizer_net import InnerProd

class NetWrapper(torch.nn.Moudle):
    def __init__(self, nets, args):
        super(NetWrapper, self).__init__()
        self.args = args

    def forward(self, batch_data, args):

        mag_mix = batch_data['mag_mix']
        mags = batch_data['mags']
        frames = batch_data['frames']
        mag_mix = mag_mix + 1e-10[0:L] 

        N = self.args.num_mix
        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # todo: what's this?
        # 0.0 warp the spectrogram
        # grid_warp = torch.from_numpy(
        #     warpgrid(B, 256, T, warp=True)).to(args.device)
        # mag_mix = F.grid_sample(mag_mix, grid_warp)
        # for n in range(N):
        #     mags[n] = F.grid_sample(mags[n], grid_warp)
        

        # 0.1 calculate loss weighting coefficient: magnitude of input mixture
        weight = torch.log1p(mag_mix)
        weight = torch.clamp(weight, 1e-3, 10)
        

        # 0.2 ground truth masks are computed after warpping!
        gt_masks = [None for n in range(N)]
        for n in range(N):
            if args['binary_mask']:
                gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()
            else:
                gt_masks[n] = mags[n] / mag_mix
                gt_masks[n].clamp_(0., 5.)

        # LOG magnitude
        log_mag_mix = torch.log(mag_mix).detach()

        # 1. forward net_sound -> BxCxHxW
        sound_features = self.net_sound(log_mag_mix)

        # 2. forward net_frame -> Bx1xC
        feat_frames = [None for n in range(N)]
        for n in range(N):
            feat_frames[n] = self.net_frame.forward_multiframe(frames[n])
            feat_frames[n] = torch.sigmoid(feat_frames[n])

        # 3. sound synthesizer
        pred_masks = [None for n in range(N)]
        for n in range(N):
            pred_masks[n] = self.net_synthesizer(feat_frames[n], sound_features)
            pred_masks[n] = torch.sigmoid(pred_masks[n])

        # 4. loss
        err = self.crit(pred_masks, gt_masks, weight).reshape(1)

        return err, \
            {'pred_masks': pred_masks, 'gt_masks': gt_masks,
             'mag_mix': mag_mix, 'mags': mags, 'weight': weight}

def build_nets():
    return (
        Unet(),
        ResnetDilated(),
        InnerProd()
    )

def weights_init(layer):
    classname = layer.__class__.__name__
    if classname.find('Conv') != -1:
        layer.weight.data.normal_(0.0, 0.001)
    elif classname.find('BatchNorm') != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        layer.weight.data.normal_(0.0, 0.0001)

def evaluate(model, loader, args):
    torch.set_grad_enabled(False)
    model.eval()
    for i, batch_data in enumerate(loader):
        error, outputs = model.forward(batch_data, args)
        error_mean = error.mean()
    print('[Eval Summary] Epoch: {}, Loss: {:.4f}')
        

def train(model, loader, optimizer, args):
    torch.set_grad_enabled(True)
    model.train()
    # todo: what's this
    # torch.cuda.synchronize()
    for i, batch_data in enumerate(loader):
        # forward pass
        model.zero_grad()
        err, _ = model.forward(batch_data, args)
        err = err.mean()

        # backward
        err.backward()
        optimizer.step()

        # print status
        if i % args['train_print_interval'] == 0:
            print('Epoch: [{}][{}/{}]'format(0,i,0))

def calc_metrics():
    pass

def checkpoint():
    pass

def calc_metrics():
    pass

if __name__ == '__main__':
    args = {
        'mix_num': 2,
        'train_print_interval': 20
    }
    nets = build_nets()
    model = NetWrapper(nets)