import torch
import torchvision
import torch.nn.functional as F

from models.audio_net import Unet
from models.vision_net import ResnetDilate
from models.synthesizer_net import SynthesizerNet
from models.net_wrapper import NetWrapper
from dataset.SolosMixDataset import SolosMixDataset

def build_nets():
    return (
        Unet(),
        ResnetDilate(),
        SynthesizerNet()
    )

def build_optimizer(nets):
    (net_sound, net_frame, net_synthesizer) = nets
    param_groups = [{'params': net_sound.parameters(), 'lr': 1e-3},
                    {'params': net_synthesizer.parameters(), 'lr': 1e-3},
                    {'params': net_frame.features.parameters(), 'lr': 1e-4},
                    {'params': net_frame.fc.parameters(), 'lr': 1e-3}]
    return torch.optim.SGD(param_groups, momentum=0.9, weight_decay=1e-4)

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
    error_history = []
    for i, batch_data in enumerate(loader):
        error, outputs = model.forward(batch_data, args)
        error_history.append(error.mean())
    return torch.mean(torch.stack(error_history))
        

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
        if i % args['print_interval_batch'] == 0:
            print('  Batch: [{}/{}], size={}'.format(i, len(loader), loader.batch_size))

def calc_metrics():
    pass


def checkpoint(nets, history, args):
    (net_sound, net_frame, net_synthesizer) = nets
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'
    path = args['ckeckpoint_path']

    torch.save(history,
               '{}/history_{}'.format(path, suffix_latest))
    torch.save(net_sound.state_dict(),
               '{}/sound_{}'.format(path, suffix_latest))
    torch.save(net_frame.state_dict(),
               '{}/frame_{}'.format(path, suffix_latest))
    torch.save(net_synthesizer.state_dict(),
               '{}/synthesizer_{}'.format(path, suffix_latest))

    if min(history)==history[-1]:
        torch.save(net_sound.state_dict(),
                   '{}/sound_{}'.format(path, suffix_best))
        torch.save(net_frame.state_dict(),
                   '{}/frame_{}'.format(path, suffix_best))
        torch.save(net_synthesizer.state_dict(),
                   '{}/synthesizer_{}'.format(path, suffix_best))
    


if __name__ == '__main__':
    args = {
        # general
        'mode': 'train',
        'seed': None,
        'mix_num': 2,
        'split': 'train',
        'batch_size': 6,#80
        'workers': 12,
        'print_interval_batch': 1,
        'evaluate_interval_epoch': 1,
        'num_epoch': 100,
        'ckeckpoint_path': 'ckpt/',
        'device': torch.device('cpu'),
        # dataset
        'train_sample_list_path': 'data/train.csv',
        'train_samples_num': 256,
        'validation_sample_list_path': 'data/val.csv',
        'validation_samples_num': 40,
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
        'stft_hop': 256,
    }


    # nets
    nets = build_nets()
    model = NetWrapper(nets, args).to(args['device'])

    # dataset and loader
    dataset_train = SolosMixDataset(args, 'train')
    dataset_validation = SolosMixDataset(args, 'validation')
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args['batch_size'],
        shuffle=True,
        # num_workers=args['workers'],
        drop_last=True)
    loader_validation = torch.utils.data.DataLoader(
        dataset_validation,
        batch_size=args['batch_size'],
        shuffle=False,
        # num_workers=args['workers'],
        drop_last=False)

    # optimizer
    optimizer = build_optimizer(nets)

    if args['mode'] == 'evaluate':
        # evaluate mode
        loss = evaluate(model, loader_validation, args)
        print('[Eval] Epoch 0, Loss: {:.4f}'.format(loss))
    elif args['mode'] == 'train':
        # train mode
        epoch_iters = len(dataset_train)
        print('1 Epoch = {} iters'.format(epoch_iters))
        loss_history = []
        for epoch in range(args['num_epoch']):
            print('Epoch {}'.format(epoch+1))
            train(model, loader_train, optimizer, args)
            if epoch % args['evaluate_interval_epoch'] == 0:
                loss = evaluate(model, loader_validation, args)
                loss_history.append(loss)
                print('[Eval] Epoch {}, Loss: {:.4f}'.format(epoch, loss))
                checkpoint(nets, loss_history, args)