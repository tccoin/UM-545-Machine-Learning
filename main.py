import torch
import torchvision
import torch.nn.functional as F

from models.audio_net import Unet
from models.vision_net import ResnetDilated
from models.synthesizer_net import InnerProd
from models.net_wrapper import NetWrapper

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
            print('Epoch: [{}][{}/{}]', format(0,i,0))

def calc_metrics():
    pass


def checkpoint(ckpt, best_err, nets, history, epoch, args):
    print('Saving checkpoints at {} epochs.'.format(epoch))
    (net_sound, net_frame, net_synthesizer) = nets
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'

    torch.save(history,
               '{}/history_{}'.format(ckpt, suffix_latest))
    torch.save(net_sound.state_dict(),
               '{}/sound_{}'.format(ckpt, suffix_latest))
    torch.save(net_frame.state_dict(),
               '{}/frame_{}'.format(ckpt, suffix_latest))
    torch.save(net_synthesizer.state_dict(),
               '{}/synthesizer_{}'.format(ckpt, suffix_latest))

    cur_err = history['val']['err'][-1]
    if cur_err < best_err:
        best_err = cur_err
        torch.save(net_sound.state_dict(),
                   '{}/sound_{}'.format(ckpt, suffix_best))
        torch.save(net_frame.state_dict(),
                   '{}/frame_{}'.format(ckpt, suffix_best))
        torch.save(net_synthesizer.state_dict(),
                   '{}/synthesizer_{}'.format(ckpt, suffix_best))


if __name__ == '__main__':
    args = {
        'mix_num': 2,
        'train_print_interval': 20,
        
    }


    nets = build_nets()
    model = NetWrapper(nets)
