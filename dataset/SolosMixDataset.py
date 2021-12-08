import os
import random
import csv
import numpy as np
import torch
import torch.utils.data as torchdata
from torchvision import transforms
import torchaudio
from PIL import Image
import librosa

class SolosMixDataset():
    def __init__(self, args):

        self.args = args
        self.mode = args['mode']
        mode_ = args['mode']+'_'

        # set seed
        if args['seed']:
            random.seed(args['seed'])

        # load samples
        sample_list_path = args[mode_+'sample_list_path']
        self.sample_list = self._load_sample_list(sample_list_path)
        self.sample_list *= args[mode_+'samples_num'] // len(self.sample_list) + 1
        random.shuffle(self.sample_list)
        self.sample_list = self.sample_list[0: args[mode_+'samples_num']]
        samples_num = len(self.sample_list)
        assert samples_num > 0
        print('number of samples: {}'.format(samples_num))
    
    def _load_sample_list(path):
        sample_list = []
        for row in csv.reader(open(path, 'r'), delimiter=','):
            if len(row) < 2:
                continue
            sample_list.append(row)
        return sample_list

    def _init_transform(self):
        x_mean, x_std = self._compute_train_statistics()
        if self.mode == 'train':
            self.transform_list = [
                transforms.Scale(self.args['frame_size']*1.2),
                transforms.RandomCrop(self.args['frame_size']),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(x_mean, x_std)
            ]
        else:
            self.transform_list = [
                transforms.Scale(self.args['frame_size']),
                transforms.CenterCrop(self.args['frame_size']),
                transforms.ToTensor(),
                transforms.Normalize(x_mean, x_std)
            ]

    def _compute_train_statistics():
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    def _init_video_transform(self):
        transform_list = []
        # ???
        self.vid_transform = transforms.Compose(transform_list)

    def _load_frames(self, paths):
        frames = []
        for path in paths:
            frames.append(self._load_frame(path))
        frames = self.vid_transform(frames)
        return frames

    def _load_frame(self, path):
        img = Image.open(path).convert('RGB')
        return img

    def _stft(self, audio):
        spec = librosa.stft(
            audio, n_fft=self.args['stft_frame'], hop_length=self.args['stft_hop'])
        amp = np.abs(spec)
        phase = np.angle(spec)
        return torch.from_numpy(amp), torch.from_numpy(phase)

    def _load_audio_file(self, path):
        if path.endswith('.mp3'):
            audio_raw, rate = torchaudio.load(path)
            audio_raw = audio_raw.numpy().astype(np.float32)

            # range to [-1, 1]
            audio_raw *= (2.0**-31)

            # convert to mono
            if audio_raw.shape[1] == 2:
                audio_raw = (audio_raw[:, 0] + audio_raw[:, 1]) / 2
            else:
                audio_raw = audio_raw[:, 0]
        else:
            audio_raw, rate = librosa.load(path, sr=None, mono=True)

        return audio_raw, rate

    def _load_audio(self, path, center_timestamp, nearest_resample=False):
        audio = np.zeros(self.args['audio_length'], dtype=np.float32)

        # silent
        if path.endswith('silent'):
            return audio

        # load audio
        audio_raw, rate = self._load_audio_file(path)

        # repeat if audio is too short
        audio_sec = 1. * self.args['audio_length'] / self.args['audio_rate']
        if audio_raw.shape[0] < rate * audio_sec:
            n = int(rate * audio_sec / audio_raw.shape[0]) + 1
            audio_raw = np.tile(audio_raw, n)

        # resample
        if rate > self.args['audio_rate']:
            # print('resmaple {}->{}'.format(rate, self.audRate))
            if nearest_resample:
                audio_raw = audio_raw[::rate//self.args['audio_rate']]
            else:
                audio_raw = librosa.resample(
                    audio_raw, rate, self.args['audio_rate'])

        # crop N seconds
        len_raw = audio_raw.shape[0]
        center = int(center_timestamp * self.args['audio_rate'])
        start = max(0, center - self.args['audio_length'] // 2)
        end = min(len_raw, center + self.args['audio_length'] // 2)

        audio[self.args['audio_length']//2-(center-start): self.args['audio_length']//2+(end-center)] = \
            audio_raw[start:end]

        # randomize volume
        if self.args['split'] == 'train':
            scale = random.random() + 0.5     # 0.5-1.5
            audio *= scale
        audio[audio > 1.] = 1.
        audio[audio < -1.] = -1.

        return audio

    def _mix_n_and_stft(self, audios):
        N = len(audios)
        mags = [None for n in range(N)]

        # mix
        for n in range(N):
            audios[n] /= N
        audio_mix = np.asarray(audios).sum(axis=0)

        # STFT
        amp_mix, phase_mix = self._stft(audio_mix)
        for n in range(N):
            ampN, _ = self._stft(audios[n])
            mags[n] = ampN.unsqueeze(0)

        # to tensor
        # audio_mix = torch.from_numpy(audio_mix)
        for n in range(N):
            audios[n] = torch.from_numpy(audios[n])

        return amp_mix.unsqueeze(0), mags, phase_mix.unsqueeze(0)

    def dummy_mix_data(self, N):
        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        mags = [None for n in range(N)]

        HS = self.args['stft_frame'] // 2 + 1
        WS = (self.args['audio_length'] + 1) // self.args['stft_hop']
        amp_mix = torch.zeros(1, HS, WS)
        phase_mix = torch.zeros(1, HS, WS)

        for n in range(N):
            frames[n] = torch.zeros(
                3, self.args['frames_per_video'], self.args['frame_size'], self.args['frame_size'])
            audios[n] = torch.zeros(self.args['audio_length'])
            mags[n] = torch.zeros(1, HS, WS)

        return amp_mix, mags, frames, audios, phase_mix

    def __getitem__(self, index):
        N = self.args['mix_num']
        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        infos = [[] for n in range(N)]
        path_frames = [[] for n in range(N)]
        path_audios = ['' for n in range(N)]
        center_frames = [0 for n in range(N)]

        # choose first sample
        infos[0] = self.sample_list[index]

        # sample other videos
        if not self.args['split'] == 'train':
            random.seed(index)
        for n in range(1, N):
            indexN = random.randint(0, len(self.sample_list)-1)
            infos[n] = self.sample_list[indexN]

        # choose frames
        idx_margin = max(int(self.args['frame_rate'] * 8),
                         (self.args['frames_per_video'] // 2) * self.args['frames_stride'])
        for n, infoN in enumerate(infos):
            path_audioN, path_frameN, count_framesN = infoN

            if self.args['split'] == 'train':
                # random, not to sample start and end n-frames
                center_frameN = random.randint(idx_margin+1, int(count_framesN)-idx_margin)
            else:
                center_frameN = int(count_framesN) // 2
            center_frames[n] = center_frameN

            # absolute frame/audio paths
            for i in range(self.args['frames_per_video']):
                idx_offset = (
                    i - self.args['frames_per_video'] // 2) * self.args['frames_stride']
                path_frames[n].append(
                    os.path.join(path_frameN,'{:06d}.jpg'.format(center_frameN + idx_offset)))
            path_audios[n] = path_audioN

        # load frames and audios, STFT
        try:
            for n, infoN in enumerate(infos):
                frames[n] = self._load_frames(path_frames[n])
                # jitter audio
                center_timeN = (center_frames[n] - 0.5) / self.args['frame_rate']
                audios[n] = self._load_audio(path_audios[n], center_timeN)
            mag_mix, mags, phase_mix = self._mix_n_and_stft(audios)

        except Exception as e:
            print('Failed loading frame/audio: {}'.format(e))
            # create dummy data
            mag_mix, mags, frames, audios, phase_mix = self.dummy_mix_data(N)

        ret_dict = {'mag_mix': mag_mix, 'frames': frames, 'mags': mags}
        if self.args['split'] != 'train':
            ret_dict['audios'] = audios
            ret_dict['phase_mix'] = phase_mix
            ret_dict['infos'] = infos

        return ret_dict


if __name__ == '__main__':
    args = {
        # general
        'mode': 'train',
        'seed': None,
        'mix_num': 2,
        'split': 'train',
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
