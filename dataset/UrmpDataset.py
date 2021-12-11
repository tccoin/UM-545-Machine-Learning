import matplotlib.pyplot as plt
import os
import random
import csv
import numpy as np
import torch
import torch.utils.data as torchdata
from torchvision import transforms
from PIL import Image
import librosa
import warnings
warnings.filterwarnings('ignore')


def show(*imgs):
    '''
     input imgs can be single or multiple tensor(s), this function uses matplotlib to visualize.
     Single input example:
     show(x) gives the visualization of x, where x should be a torch.Tensor
        if x is a 4D tensor (like image batch with the size of b(atch)*c(hannel)*h(eight)*w(eight), this function splits x in batch dimension, showing b subplots in total, where each subplot displays first 3 channels (3*h*w) at most. 
        if x is a 3D tensor, this function shows first 3 channels at most (in RGB format)
        if x is a 2D tensor, it will be shown as grayscale map

     Multiple input example:      
     show(x,y,z) produces three windows, displaying x, y, z respectively, where x,y,z can be in any form described above.
    '''
    img_idx = 0
    for img in imgs:
        img_idx += 1
        plt.figure(img_idx)
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()

            if img.dim() == 4:  # 4D tensor
                bz = img.shape[0]
                c = img.shape[1]
                if bz == 1 and c == 1:  # single grayscale image
                    img = img.squeeze()
                elif bz == 1 and c == 3:  # single RGB image
                    img = img.squeeze()
                    img = img.permute(1, 2, 0)
                elif bz == 1 and c > 3:  # multiple feature maps
                    img = img[:, 0:3, :, :]
                    img = img.permute(0, 2, 3, 1)[:]
                    print(
                        'warning: more than 3 channels! only channels 0,1,2 are preserved!')
                elif bz > 1 and c == 1:  # multiple grayscale images
                    img = img.squeeze()
                elif bz > 1 and c == 3:  # multiple RGB images
                    img = img.permute(0, 2, 3, 1)
                elif bz > 1 and c > 3:  # multiple feature maps
                    img = img[:, 0:3, :, :]
                    img = img.permute(0, 2, 3, 1)[:]
                    print(
                        'warning: more than 3 channels! only channels 0,1,2 are preserved!')
                else:
                    raise Exception("unsupported type!  " + str(img.size()))
            elif img.dim() == 3:  # 3D tensor
                bz = 1
                c = img.shape[0]
                if c == 1:  # grayscale
                    img = img.squeeze()
                elif c == 3:  # RGB
                    img = img.permute(1, 2, 0)
                else:
                    raise Exception("unsupported type!  " + str(img.size()))
            elif img.dim() == 2:
                pass
            else:
                raise Exception("unsupported type!  "+str(img.size()))

            img = img.numpy()  # convert to numpy
            img = img.squeeze()
            if bz == 1:
                plt.imshow(img, cmap='gray')
                # plt.colorbar()
                # plt.show()
            else:
                for idx in range(0, bz):
                    plt.subplot(
                        int(bz**0.5), int(np.ceil(bz/int(bz**0.5))), int(idx+1))
                    plt.imshow(img[idx], cmap='gray')

        else:
            raise Exception("unsupported type:  "+str(type(img)))


class UrmpDataset(torchdata.Dataset):
    def __init__(self, args, split='test'):

        self.args = args
        self.split = split
        split_ = split+'_'

        # init transform
        self._init_transform()

        # set seed
        if args['seed']:
            random.seed(args['seed'])

        # load samples
        sample_list_path = args[split_+'sample_list_path']
        self.sample_list = self._load_sample_list(sample_list_path)
        self.sample_list *= args[split_ +
                                 'samples_num'] // len(self.sample_list) + 1
        random.shuffle(self.sample_list)

        self.sample_list = self.sample_list[0: args[split_+'samples_num']]
        samples_num = len(self.sample_list)
        assert samples_num > 0
        print('number of samples: {}'.format(samples_num))

    def _load_sample_list(self, path):
        sample_list = []
        for row in csv.reader(open(path, 'r'), delimiter=','):
            if len(row) < 2:
                continue
            sample_list.append(row)
        return sample_list

    def __len__(self):
        return len(self.sample_list)

    def _init_transform(self):
        x_mean, x_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        if self.split == 'train':
            transform_list = [
                transforms.Resize(int(self.args['frame_size']*1.2)),
                transforms.RandomCrop(self.args['frame_size']),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(x_mean, x_std)
            ]
        else:
            transform_list = [
                transforms.Scale(self.args['frame_size']),
                transforms.CenterCrop(self.args['frame_size']),
                transforms.ToTensor(),
                transforms.Normalize(x_mean, x_std)
            ]
        self.img_transform = transforms.Compose(transform_list)

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
        audio_raw, rate = librosa.load(path, sr=None, mono=True)
        return audio_raw, rate

    def _load_audio(self, path, center_timestamp, nearest_resample=False):
        audio = np.zeros(self.args['audio_length'], dtype=np.float32)

        # silent
        if path.endswith('silent'):
            return audio

        # load audio
        audio_raw, rate = self._load_audio_file(path)

        # print('raw_audio length: {}/{}/{}'.format(len(audio_raw), center_timestamp*2,len(audio_raw)/center_timestamp/2))
        # repeat if audio is too short
        audio_sec = 1. * self.args['audio_length'] / self.args['audio_rate']
        if audio_raw.shape[0] < rate * audio_sec:
            n = int(rate * audio_sec / audio_raw.shape[0]) + 1
            audio_raw = np.tile(audio_raw, n)

        # resample
        if rate > self.args['audio_rate']:
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
        if self.split == 'train':
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
        info = self.sample_list[index]
        N = 5
        N_ = info[0].count('_')-1
        frames = [torch.zeros(
            3, self.args['frames_per_video'], self.args['frame_size'], self.args['frame_size']) for n in range(N)]
        audios = [np.zeros(self.args['audio_length'], dtype=np.single)
                  for n in range(N)]

        # load frames and audios, STFT
        path_audio, path_frame, frame_num = info
        path_frames = [path_frame+'/19.jpg',
                       path_frame+'/22.jpg',
                       path_frame+'/25.jpg']
        original_frames = []
        for i, path in enumerate(path_frames):
            original_frames.append(np.asarray(self._load_frame(path)))
        # show(torch.stack([transforms.ToTensor()(x) for x in original_frames]))
        # plt.show()
        original_frames = np.array(original_frames)
        T, H, W, C = original_frames.shape
        for n in range(N_):
            frames[n] = original_frames[:,
                                        :,
                                        W//N_*n:W//N_*(n+1), :]
            frame_T = []
            for i in range(T):
                img = Image.fromarray(np.uint8(frames[n][i]))
                frame_T.append(self.img_transform(img))
            frames[n] = torch.stack(frame_T)
            # show(frames[n])

        for n in range(N_):
            center_timeN = int(frame_num) // 2
            audio_path = path_audio[:-4]+'_{}.wav'.format(n)
            audios[n] = self._load_audio(audio_path, center_timeN)
        mag_mix, mags, phase_mix = self._mix_n_and_stft(audios)

        # except Exception as e:
        #     print('Failed loading frame/audio: {}'.format(e))
        #     # create dummy data
        #     mag_mix, mags, frames, audios, phase_mix = self.dummy_mix_data(N)

        ret_dict = {'mag_mix': mag_mix, 'frames': frames, 'mags': mags}
        if self.split != 'train':
            ret_dict['audios'] = audios
            ret_dict['phase_mix'] = phase_mix

        return ret_dict


if __name__ == '__main__':
    args = {
        # general
        'mode': 'train',
        'seed': None,
        'mix_num': 2,
        'batch_size': 10,
        'workers': 12,
        # dataset
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
        'log_freq': 1,
        'stft_frame': 1022,
        'stft_hop': 256
    }
    dataset_test = UrmpDataset(args, 'test')
    batch_data = dataset_test[0]
    for i, batch_data in enumerate(batch_data):
        print(i)
        break
