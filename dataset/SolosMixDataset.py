import os
import random
import csv
import numpy as np
import torch
import torch.utils.data as torchdata
from torchvision import transforms
import torchaudio

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

    def __getitem__(self, index):
        N = self.args['mix_num']

        # choose first sample

        # sample other videos

        # choose frames

        # load frames and audios, STFT

        return None


if __name__ == '__main__':
    args = {
        # general
        'mode': 'train',
        'seed': None,
        'mix_num': 2,
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