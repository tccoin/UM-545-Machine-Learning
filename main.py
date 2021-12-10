import torch
import torchvision
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import librosa

from models.audio_net import Unet
from models.vision_net import ResnetDilate
from models.synthesizer_net import SynthesizerNet
from models.net_wrapper import NetWrapper
from models.metrics import AverageMeter
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
    for i, batch_data in enumerate(loader):
        # forward pass
        model.zero_grad()
        err, _ = model.forward(batch_data, args)
        err = err.mean()

        # backward
        err.backward()
        optimizer.step()

        # print status
        args['writer'].add_scalar('Loss/train', err, args['current_epoch']*len(loader)+i)
        if i % args['print_interval_batch'] == 0:
            print('  Batch: [{}/{}], size={}'.format(i, len(loader), loader.batch_size))

            
            
# audio signal reconstruction and final model evaluations
def istft_reconstruction(mag, phase, hop_length=256):
    spec = mag.astype(np.complex) * np.exp(1j*phase)
    wav = librosa.istft(spec, hop_length=hop_length)
    return np.clip(wav, -1., 1.)
               
            
def calc_metrics():
    # meters
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    # fetch data and predictions
    mag_mix = batch_data['mag_mix']
    phase_mix = batch_data['phase_mix']
    audios = batch_data['audios']

    pred_masks_ = outputs['pred_masks']

    # unwarp log scale
    N = args.num_mix
    B = mag_mix.size(0)
    pred_masks_linear = [None for n in range(N)]
    for n in range(N):
        if args.log_freq:
            grid_unwarp = torch.from_numpy(
                warpgrid(B, args.stft_frame//2+1, pred_masks_[0].size(3), warp=False)).to(args.device)
            pred_masks_linear[n] = F.grid_sample(pred_masks_[n], grid_unwarp)
        else:
            pred_masks_linear[n] = pred_masks_[n]

    # convert into numpy
    mag_mix = mag_mix.numpy()
    phase_mix = phase_mix.numpy()
    for n in range(N):
        pred_masks_linear[n] = pred_masks_linear[n].detach().cpu().numpy()

        # threshold if binary mask
        if args.binary_mask:
            pred_masks_linear[n] = (pred_masks_linear[n] > args.mask_thres).astype(np.float32)

    # loop over each sample
    for j in range(B):
        # save mixture
        mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)

        # save each component
        preds_wav = [None for n in range(N)]
        for n in range(N):
            # Predicted audio recovery
            pred_mag = mag_mix[j, 0] * pred_masks_linear[n][j, 0]
            preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)

        # separation performance computes
        L = preds_wav[0].shape[0]
        gts_wav = [None for n in range(N)]
        valid = True
        for n in range(N):
            gts_wav[n] = audios[n][j, 0:L].numpy()
            valid *= np.sum(np.abs(gts_wav[n])) > 1e-5
            valid *= np.sum(np.abs(preds_wav[n])) > 1e-5
        if valid:
            sdr, sir, sar, _ = bss_eval_sources(np.asarray(gts_wav), np.asarray(preds_wav), compute_permutation=False)
            sdr_mix, _, _, _ = bss_eval_sources(np.asarray(gts_wav), np.asarray([mix_wav[0:L] for n in range(N)]), compute_permutation=False)
            
            sdr_mix_meter.update(sdr_mix.mean())
            sdr_meter.update(sdr.mean())
            sir_meter.update(sir.mean())
            sar_meter.update(sar.mean())

    return [sdr_mix_meter.average(), sdr_meter.average(), sir_meter.average(),sar_meter.average()]


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
        'batch_size': 24,
        'workers': 12,
        'print_interval_batch': 1,
        'evaluate_interval_epoch': 1,
        'num_epoch': 100,
        'ckeckpoint_path': 'ckpt/',
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

    args['device'] = torch.device('cuda')
    args['writer'] = SummaryWriter()
    args['current_epoch'] = 0

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
            args['current_epoch'] = epoch
            print('Epoch {}'.format(epoch+1))
            train(model, loader_train, optimizer, args)
            if epoch % args['evaluate_interval_epoch'] == 0:
                loss = evaluate(model, loader_validation, args)
                loss_history.append(loss)
                args['writer'].add_scalar('Loss/validation', loss, epoch)
                print('[Eval] Epoch {}, Loss: {:.4f}'.format(epoch+1, loss))
                checkpoint(nets, loss_history, args)