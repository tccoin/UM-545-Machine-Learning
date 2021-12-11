import torch
import torchvision
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import librosa
import numpy as np

from models.audio_net import Unet
from models.vision_net import ResnetDilate
from models.synthesizer_net import SynthesizerNet
from net_wrapper import NetWrapper
from models.metrics import AverageMeter
from dataset.SolosMixDataset import SolosMixDataset
from dataset.UrmpDataset import UrmpDataset
from utils import save_checkpoint, load_checkpoint, calc_metrics
import random


def build_nets(args):
    return (
        Unet(),
        ResnetDilate(pretrained=args['resnet_pretrained']),
        SynthesizerNet()
    )


def build_optimizer(nets, args):
    (net_sound, net_frame, net_synthesizer) = nets
    param_groups = [{'params': net_sound.parameters(), 'lr': args['net_sound_lr']},
                    {'params': net_synthesizer.parameters(
                    ), 'lr': args['net_synthesizer_lr']},
                    {'params': net_frame.features.parameters(),
                     'lr': args['net_frames_lr']},
                    {'params': net_frame.fc.parameters(), 'lr': args['net_frames_lr']}]
    return torch.optim.SGD(param_groups, momentum=0.9, weight_decay=args['weight_decay'])


def weights_init(layer):
    classname = layer.__class__.__name__
    if classname.find('Conv') != -1:
        layer.weight.data.normal_(0.0, 0.001)
    elif classname.find('BatchNorm') != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        layer.weight.data.normal_(0.0, 0.0001)


def test(model, loader, args):
    torch.set_grad_enabled(False)
    model.eval()
    error_history = []
    metrics_list = []  # sdr_mix, sdr, sir, sar

    for i, batch_data in enumerate(loader):
        error, outputs = model.forward(batch_data, args)
        error_history.append(error.mean())
        metrics_list.append(calc_metrics(batch_data, outputs, args))
    # record loss
    metrics_list = np.array(metrics_list)
    metrics = np.mean(metrics_list, axis=0)
    epoch_num = args['current_epoch']+1
    args['writer'].add_scalar('Metrics/sdr_mix', metrics[0], epoch_num)
    args['writer'].add_scalar('Metrics/sdr', metrics[1], epoch_num)
    args['writer'].add_scalar('Metrics/sir', metrics[2], epoch_num)
    args['writer'].add_scalar('Metrics/sar', metrics[3], epoch_num)
    args['history']['metrics'].append((args['current_epoch'], metrics))
    loss = torch.mean(torch.stack(error_history))
    args['history']['test_loss'].append((args['current_epoch'], loss))
    args['writer'].add_scalar('Loss/test', loss, epoch_num)
    print('[Test] Epoch {}, Loss: {:.4f}'.format(epoch_num, loss))


def evaluate(model, loader, args):
    torch.set_grad_enabled(False)
    model.eval()
    error_history = []

    for i, batch_data in enumerate(loader):
        error, outputs = model.forward(batch_data, args)
        error_history.append(error.mean())
    # record loss
    epoch_num = args['current_epoch']+1
    loss = torch.mean(torch.stack(error_history))
    args['history']['validation_loss'].append((args['current_epoch'], loss))
    args['writer'].add_scalar('Loss/validation', loss, epoch_num)
    print('[Eval] Epoch {}, Loss: {:.4f}'.format(epoch_num, loss))
    # weight histogram
    weights_list = [
        ('net_frame.fc.bias', model.net_frame.fc.bias),
        ('net_frame.fc.weight', model.net_frame.fc.weight),
        ('net_frame.layer0.weight', model.net_frame.features[0].weight),
        ('net_sound.layer0.weight', model.net_sound.unet.model[0].weight),
        ('net_sound.layer2.weight',
         model.net_sound.unet.model[1].model[1].weight),

        ('net_sound.layer4.weight', model.net_sound.unet.model[4].weight),
        ('net_thethesizer.scale', model.net_synthesizer.scale),
        ('net_thethesizer.bias', model.net_synthesizer.bias),
    ]
    for weights in weights_list:
        args['writer'].add_histogram(weights[0], weights[1], epoch_num)


def train(model, loader, optimizer, args):
    torch.set_grad_enabled(True)
    model.train()
    for i, batch_data in enumerate(loader):
        # forward pass
        model.zero_grad()
        err, _ = model.forward(batch_data, args)
        err = err.mean()

        # backward
        err.backward()
        optimizer.step()

        # record loss
        total_batch_number = args['current_epoch']*len(loader)+i
        args['history']['train_loss'].append((total_batch_number, err))
        args['writer'].add_scalar('Loss/train', err, total_batch_number)
        if i % args['print_interval_batch'] == 0:
            print('  Batch: [{}/{}], size={}, loss={:.4f}'.format(i+1,
                  len(loader), loader.batch_size, err))


if __name__ == '__main__':
    args = {
        # general
        'mode': 'test',
        'seed': None,
        'batch_size': 24,
        'workers': 24,
        'print_interval_batch': 1,
        'evaluate_interval_epoch': 5,
        'ckeckpoint_path': 'ckpt/',
        # training
        'num_epoch': 100,
        'net_sound_lr': 1e-3,
        'net_frames_lr': 1e-3,
        'net_synthesizer_lr': 1e-3,
        'net_synthesizer_lr': 1e-3,
        'weight_decay': 1e-3,
        'resnet_pretrained': True,
        # dataset
        'mix_num': 2,
        'train_samples_num': 256,
        'validation_samples_num': 40,
        'train_sample_list_path': 'data/train.csv',
        'validation_sample_list_path': 'data/val.csv',
        'test_sample_list_path': 'data/test.csv',
        'test_samples_num': 44,
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
        'log_freq': True,
        'stft_frame': 1022,
        'stft_hop': 256,
    }

    args['device'] = torch.device('cuda')
    args['writer'] = SummaryWriter()
    args['current_epoch'] = 0
    # args['seed'] = random.randint(0, 10000)
    args['seed'] = 6480
    args['history'] = {'train_loss': [],
                       'validation_loss': [], 'test_loss': [], 'metrics': []}

    # nets
    nets = build_nets()
    model = NetWrapper(nets, args).to(args['device'])
    optimizer = build_optimizer(nets)

    # dataset and loader
    dataset_train = SolosMixDataset(args, 'train')
    dataset_validation = SolosMixDataset(args, 'validation')
    dataset_test = UrmpDataset(args, 'test')
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
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args['batch_size'],
        shuffle=True,
        # num_workers=args['workers'],
        drop_last=True)

    # load checkpoint
    # load_checkpoint('ckpt/latest.pth', model, optimizer, args)

    if args['mode'] == 'test':
        # test mode
        test(model, loader_test, args)
    elif args['mode'] == 'train':
        # train
        epoch_iters = len(dataset_train)
        print('1 Epoch = {} iters'.format(epoch_iters))
        for epoch in range(args['current_epoch'], args['num_epoch']):
            args['current_epoch'] = epoch
            print('Epoch {}'.format(epoch+1))
            train(model, loader_train, optimizer, args)
            evaluate(model, loader_validation, args)
            if epoch % args['evaluate_interval_epoch'] == 0:
                test(model, loader_test, args)
                save_checkpoint(model, optimizer, args)
