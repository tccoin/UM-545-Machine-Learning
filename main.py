import torch
import torchvision
import torch.nn.functional as F

from models.audio_net import Unet
from models.vision_net import ResnetDilated
from models.synthesizer_net import InnerProd
from models.net_wrapper import NetWrapper
from dataset.SolosMixDataset import SolosMixDataset

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

if __name__ == '__main__':
    args = {
        # general
        'mode': 'train',
        'seed': None,
        'mix_num': 2,
        'split': 'train',
        'batch_size': 80,
        'workers': 12,
        'train_print_interval': 20,
        # dataset
        'train_sample_list_path': 'data/train.csv',
        'train_samples_num': 256,
        'val_sample_list_path': 'data/val.csv',
        'val_samples_num': 40,
        # frames
        'frame_size': 224,
        'frames_per_video': 3,
        'frames_stride': 24,
        'frame_rate': 8,
        # audio
        'audio_rate': 11025,
        'audio_length': 65535,
        'use_binary_mask': True,
        # STFT
        'log_freq': 1,
        'stft_frame': 1022,
        'stft_hop': 256
    }
    nets = build_nets()
    model = NetWrapper(nets)
    dataset = SolosMixDataset(args)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args['batch_size'],
        shuffle=True,
        num_workers=int(args['workers']),
        drop_last=True)
    for i, batch_data in enumerate(loader):
        print(i)
