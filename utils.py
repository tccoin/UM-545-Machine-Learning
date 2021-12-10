import numpy as np
import librosa
from mir_eval.separation import bss_eval_sources
import torch


def warpgrid(bs, HO, WO, warp=True):
    # meshgrid
    x = np.linspace(-1, 1, WO)
    y = np.linspace(-1, 1, HO)
    xv, yv = np.meshgrid(x, y)
    grid = np.zeros((bs, HO, WO, 2))
    grid_x = xv
    if warp:
        grid_y = (np.power(21, (yv+1)/2) - 11) / 10
    else:
        grid_y = np.log(yv * 10 + 11) / np.log(21) * 2 - 1
    grid[:, :, :, 0] = grid_x
    grid[:, :, :, 1] = grid_y
    grid = grid.astype(np.float32)
    return grid

# audio signal reconstruction and final model evaluations


def istft_reconstruction(mag, phase, hop_length=256):
    spec = mag.astype(np.complex) * np.exp(1j*phase)
    wav = librosa.istft(spec, hop_length=hop_length)
    return np.clip(wav, -1., 1.)


def calc_metrics(batch_data, outputs, args):
    # meters
    sdr_mix_meter = []
    sdr_meter = []
    sir_meter = []
    sar_meter = []

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
            pred_masks_linear[n] = (
                pred_masks_linear[n] > args.mask_thres).astype(np.float32)

    # loop over each sample
    for j in range(B):
        # save mixture
        mix_wav = istft_reconstruction(
            mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)

        # save each component
        preds_wav = [None for n in range(N)]
        for n in range(N):
            # Predicted audio recovery
            pred_mag = mag_mix[j, 0] * pred_masks_linear[n][j, 0]
            preds_wav[n] = istft_reconstruction(
                pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)

        # separation performance computes
        L = preds_wav[0].shape[0]
        gts_wav = [None for n in range(N)]
        valid = True
        for n in range(N):
            gts_wav[n] = audios[n][j, 0:L].numpy()
            valid *= np.sum(np.abs(gts_wav[n])) > 1e-5
            valid *= np.sum(np.abs(preds_wav[n])) > 1e-5
        if valid:
            sdr, sir, sar, _ = bss_eval_sources(np.asarray(
                gts_wav), np.asarray(preds_wav), compute_permutation=False)
            sdr_mix, _, _, _ = bss_eval_sources(np.asarray(gts_wav), np.asarray(
                [mix_wav[0:L] for n in range(N)]), compute_permutation=False)

            sdr_mix_meter.update(sdr_mix.mean())
            sdr_meter.update(sdr.mean())
            sir_meter.update(sir.mean())
            sar_meter.update(sar.mean())

    return [sdr_mix_meter.average(), sdr_meter.average(), sir_meter.average(), sar_meter.average()]


def save_checkpoint(model, optimizer, args):
    path_latest = args['ckeckpoint_path'] + '/latest.pth'
    path_best = args['ckeckpoint_path'] + '/best.pth'
    data = {
        'seed': args['seed'],
        'history': args['history'],
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': args['current_epoch']
    }

    torch.save(data, path_latest)
    history = args['history']
    min_loss = min(history['validation_loss'], key=lambda x: x[1])[1]
    current_loss = history['validation_loss'][-1][1]
    print('loss history: {:.4f} vs {:.4f}'.format(current_loss, min_loss))
    if current_loss == min_loss:
        print('Best model ever!')
        torch.save(data, path_best)


def load_checkpoint(path, model, optimizer, args):
    checkpoint = torch.load(path)
    args['seed'] = checkpoint['seed']
    args['history'] = checkpoint['history']
    args['current_epoch'] = checkpoint['epoch']+1
    model.load_state_dict(checkpoint['model'])
    model.eval()
    optimizer.load_state_dict(checkpoint['optimizer'])
    # record history loss
    for i, loss in args['history']['validation_loss']:
        args['writer'].add_scalar(
            'Loss/validation', loss, i+1)
    for i, loss in args['history']['train_loss']:
        args['writer'].add_scalar(
            'Loss/train', loss, i)
