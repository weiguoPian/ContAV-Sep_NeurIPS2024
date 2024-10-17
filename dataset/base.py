import random
import os
import csv
import numpy as np
import torch
import torch.utils.data as torchdata
from torchvision import transforms
import torchaudio
import librosa
from PIL import Image
import torch.nn.functional as F

from . import video_transforms as vtransforms


class BaseDataset(torchdata.Dataset):
    def __init__(self, opt, split='train'):
        # params
        self.num_frames = opt.num_frames
        self.stride_frames = opt.stride_frames
        self.frameRate = opt.frameRate
        self.imgSize = opt.imgSize
        self.audRate = opt.audRate
        self.audLen = opt.audLen
        self.audSec = 1. * self.audLen / self.audRate
        self.binary_mask = opt.binary_mask

        # STFT params
        self.log_freq = opt.log_freq
        self.stft_frame = opt.stft_frame
        self.stft_hop = opt.stft_hop
        self.HS = opt.stft_frame // 2 + 1
        self.WS = (self.audLen + 1) // self.stft_hop

        self.split = split

        # initialize video transform
        self._init_vtransform()

        # Your data root
        # self.data_root = ''
        self.data_root = '/data/home/weiguo/dataset/MIT-Music'

        self.all_category_list = ['accordion', 'bagpipe', 'bassoon', 'clarinet', 'drum', \
                                  'flute', 'piano', 'saxophone', 'tuba', 'violin', \
                                  'xylophone', 'acoustic_guitar', 'banjo', 'cello', 'electric_bass', \
                                  'erhu', 'guzheng', 'pipa', 'trumpet', 'ukulele']

        self.category_encode_dict = dict(zip(self.all_category_list, list(range(len(self.all_category_list)))))
        self.class2category_dict = dict(zip(list(range(len(self.all_category_list))), self.all_category_list))

        self.all_class_list = []
        self.all_vids_list = []

        self.category_vids_dict = np.load('./data/{}_category_vids_dict.npy'.format(self.split), allow_pickle=True).item()
        for c in self.all_category_list:
            vids_list = self.category_vids_dict[c]
            class_list = [self.category_encode_dict[c]] * len(vids_list)
            self.all_vids_list += vids_list
            self.all_class_list += class_list
        
        self.vid_class_dict = dict(zip(self.all_vids_list, self.all_class_list))

        if self.split == 'train':
            self.all_vids_list *= opt.dup_trainset
        
        self.motion_features = np.load(os.path.join(self.data_root, 'motion_features', 'motion_features_dict.npy'), allow_pickle=True).item()

        self.audio_raw_data = np.load(
            os.path.join(self.data_root, 'audio_npys_continual', '{}_vid_raw_audio_dict.npy'.format(self.split)), allow_pickle=True
        ).item()

        self.all_objects_features = np.load(
            os.path.join(self.data_root, 'objects_features', '{}_vid_objects_features_dict.npy'.format(self.split)), allow_pickle=True
        ).item()

    def __len__(self):
        return len(self.all_vids_list)

    # video transform funcs
    def _init_vtransform(self):
        transform_list = []
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.split == 'train':
            transform_list.append(vtransforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC))
            transform_list.append(vtransforms.RandomCrop(self.imgSize))
            transform_list.append(vtransforms.RandomHorizontalFlip())
        else:
            transform_list.append(vtransforms.Resize(self.imgSize, Image.BICUBIC))
            transform_list.append(vtransforms.CenterCrop(self.imgSize))

        transform_list.append(vtransforms.ToTensor())
        transform_list.append(vtransforms.Normalize(mean, std))
        transform_list.append(vtransforms.Stack())
        self.vid_transform = transforms.Compose(transform_list)

    # image transform funcs, deprecated
    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.split == 'train':
            self.img_transform = transforms.Compose([
                transforms.Scale(int(self.imgSize)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            self.img_transform = transforms.Compose([
                transforms.Scale(self.imgSize),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])


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
            audio, n_fft=self.stft_frame, hop_length=self.stft_hop)
        amp = np.abs(spec)
        phase = np.angle(spec)
        return torch.from_numpy(amp), torch.from_numpy(phase)
    
    def _load_audio_file(self, vid):
        rate = 11025
        audio_raw = self.audio_raw_data[vid]
        return audio_raw, rate

    def _load_audio(self, path, center_timestamp, nearest_resample=False):
        audio = np.zeros(self.audLen, dtype=np.float32)

        # silent
        if path.endswith('silent'):
            return audio

        # load audio
        audio_raw, rate = self._load_audio_file(path)

        # repeat if audio is too short
        if audio_raw.shape[0] < rate * self.audSec:
            n = int(rate * self.audSec / audio_raw.shape[0]) + 1
            audio_raw = np.tile(audio_raw, n)

        # resample
        if rate > self.audRate:
            # print('resmaple {}->{}'.format(rate, self.audRate))
            if nearest_resample:
                audio_raw = audio_raw[::rate//self.audRate]
            else:
                audio_raw = librosa.resample(audio_raw, rate, self.audRate)

        # crop N seconds
        len_raw = audio_raw.shape[0]
        center = int(center_timestamp * self.audRate)
        start = max(0, center - self.audLen // 2)
        end = min(len_raw, center + self.audLen // 2)

        audio[self.audLen//2-(center-start): self.audLen//2+(end-center)] = \
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
        for n in range(N):
            audios[n] /= N
        audio_mix = np.asarray(audios).sum(axis=0)

        # STFT
        amp_mix, phase_mix = self._stft(audio_mix)
        for n in range(N):
            ampN, _ = self._stft(audios[n])
            mags[n] = ampN.unsqueeze(0)
        
        # to tensor
        for n in range(N):
            audios[n] = torch.from_numpy(audios[n])
        return amp_mix.unsqueeze(0), mags, phase_mix.unsqueeze(0)

    def dummy_mix_data(self, N):
        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        mags = [None for n in range(N)]

        amp_mix = torch.zeros(1, self.HS, self.WS)
        phase_mix = torch.zeros(1, self.HS, self.WS)

        for n in range(N):
            frames[n] = torch.zeros(
                3, self.num_frames, self.imgSize, self.imgSize)
            audios[n] = torch.zeros(self.audLen)
            mags[n] = torch.zeros(1, self.HS, self.WS)
        return amp_mix, mags, frames, audios, phase_mix
