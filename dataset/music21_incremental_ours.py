import os
import random
import numpy as np
import csv
from .base import BaseDataset
import torch
import copy
import torch.nn.functional as F

class MUSICMix21IncrementalOursDataset(BaseDataset):
    def __init__(self, opt, **kwargs):
        super(MUSICMix21IncrementalOursDataset, self).__init__(opt, **kwargs)
        self.fps = opt.frameRate
        self.num_mix = opt.num_mix
        self.args = opt

        self.incremental_step = None
        self.current_step_classes = None
        self._set_incremental_step_(step=0)

        self.exemplar_class_vids_set = []
        self.exemplar_vids_set = []
    
    def _set_incremental_step_(self, step):
        self.incremental_step = step
        self._set_current_step_classes_()
        self._current_step_data_()

        if self.split == 'train':
            if self.incremental_step > 0:
                self._update_exemplars_()
                self.exemplar_vids_set *= (self.args.dup_trainset * 20)
                self.all_vids_list *= self.args.dup_trainset
                self.all_vids_list += self.exemplar_vids_set
            else:
                self.all_vids_list *= self.args.dup_trainset
    
    def _set_current_step_classes_(self):
        if self.split == 'train':
            if self.args.upper_bound:
                self.current_step_classes = np.array(range(0, self.args.class_num_per_step * (self.incremental_step + 1)))
            else:
                self.current_step_classes = np.array(range(self.args.class_num_per_step * self.incremental_step, self.args.class_num_per_step * (self.incremental_step + 1)))
        else:
            self.current_step_classes = np.array(range(0, self.args.class_num_per_step * (self.incremental_step + 1)))
    
    def _current_step_data_(self):
        all_current_data_vids = []
        for class_idx in self.current_step_classes:
            all_current_data_vids += self.category_vids_dict[self.class2category_dict[class_idx]]
        self.all_vids_list = copy.deepcopy(all_current_data_vids)

    def _update_exemplars_(self):
        new_memory_classes = range((self.incremental_step - 1) * self.args.class_num_per_step, self.incremental_step * self.args.class_num_per_step)
        # exemplar_num_per_class = self.args.memory_size // (self.incremental_step * self.args.class_num_per_step)
        exemplar_num_per_class = self.args.exemplar_num_per_class

        if exemplar_num_per_class == 0:
            exemplar_num_per_class = 1

        new_memory_class_exemplars = self._init_new_memory_class_exemplars_(new_memory_classes, exemplar_num_per_class)

        if self.incremental_step == 1:
            self.exemplar_class_vids_set += new_memory_class_exemplars
        else:
            for i in range(len(self.exemplar_class_vids_set)):
                self.exemplar_class_vids_set[i] = self.exemplar_class_vids_set[i][:exemplar_num_per_class]
            self.exemplar_class_vids_set += new_memory_class_exemplars
        
        self.exemplar_vids_set = np.array(self.exemplar_class_vids_set).reshape(-1).tolist()
        self.exemplar_vids_set = [vid for vid in self.exemplar_vids_set if vid is not None]

        # self.exemplar_vids_set = self.exemplar_vids_set[:self.args.memory_size]


    def _init_new_memory_class_exemplars_(self, new_memory_classes, exemplar_num_per_class):
        new_memory_class_exemplars = []
        for i in new_memory_classes:
            class_vids = self.category_vids_dict[self.class2category_dict[i]]
            class_exemplar = random.sample(class_vids, min(len(class_vids), exemplar_num_per_class))
            if len(class_vids) < exemplar_num_per_class:
                class_exemplar += [None for j in range(exemplar_num_per_class-len(class_vids))]
            new_memory_class_exemplars.append(class_exemplar)
        return new_memory_class_exemplars
        




    def __getitem__(self, index):
        N = self.num_mix
        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        infos = [[] for n in range(N)] #[[], [], [], []]

        center_frames = [0 for n in range(N)]

        class_list = []

        motions = [None for n in range(N)]

        vid_1 = self.all_vids_list[index]

        if self.split != 'train':
            random.seed(index)

        vid_2 = self.all_vids_list[random.randint(0, len(self.all_vids_list)-1)]

        category_1 = self.class2category_dict[self.vid_class_dict[vid_1]]
        category_2 = self.class2category_dict[self.vid_class_dict[vid_2]]
        
        audio_1_path = os.path.join(self.data_root, 'audio_11025', category_1, '{}.wav'.format(vid_1))
        audio_2_path = os.path.join(self.data_root, 'audio_11025', category_2, '{}.wav'.format(vid_2))

        frames_1_root = os.path.join(self.data_root, 'objects', category_1, vid_1)
        frames_2_root = os.path.join(self.data_root, 'objects', category_2, vid_2)

        count_framesN_1 = len(os.listdir(frames_1_root))
        count_framesN_2 = len(os.listdir(frames_2_root))

        class_list.append(self.vid_class_dict[vid_1])
        class_list.append(self.vid_class_dict[vid_2])

        frames_roots = [frames_1_root, frames_2_root]

        infos = [(audio_1_path, frames_1_root, count_framesN_1), (audio_2_path, frames_2_root, count_framesN_2)]
        # infos = [(audio_1_path, count_framesN_1), (audio_2_path, count_framesN_2)]
        
        count_framesN_list = [count_framesN_1, count_framesN_2]

        vids = [vid_1, vid_2]

        frames_features = []

        # select frames
        idx_margin = max(
            int(self.fps * 8), (self.num_frames // 2) * self.stride_frames)
        # for n, infoN in enumerate(infos):
        for n, vid in enumerate(vids):
            # path_audioN, path_frameN, count_framesN = infoN
            count_framesN = count_framesN_list[n]
            vid_frames_features = []

            if self.split == 'train':
                # random, not to sample start and end n-frames
                if idx_margin+1 < int(count_framesN)-idx_margin:
                    center_frameN = random.randint(idx_margin+1, int(count_framesN)-idx_margin)
                else:
                    center_frameN = random.randint(int(count_framesN)-idx_margin, idx_margin+1)
            else:
                center_frameN = int(count_framesN) // 2
                single_video_frames_paths = []
            center_frames[n] = center_frameN

            # absolute frame/audio paths
            for i in range(self.num_frames):
                idx_offset = (i - self.num_frames // 2) * self.stride_frames

                vid_frames_features.append(
                    self.all_objects_features[vid][center_frameN + idx_offset - 1]
                )
                if self.split != 'train':
                    single_frame_path = os.path.join(frames_roots[n], '{:06d}.jpg'.format(center_frameN + idx_offset))
                    single_video_frames_paths.append(single_frame_path)

            vid_frames_features = torch.Tensor(vid_frames_features).permute(1, 0)
            vid_frames_features = F.adaptive_max_pool1d(vid_frames_features, 1).detach().squeeze(1)
            # path_audios[n] = path_audioN
            frames_features.append(vid_frames_features)

            center_timeN = (center_frameN - 0.5) / self.fps
            audios[n] = self._load_audio(vid, center_timeN)

            motions[n] = torch.Tensor(self.motion_features[vid])

            if self.split != 'train':
                frames[n] = self._load_frames(single_video_frames_paths)

        mag_mix, mags, phase_mix = self._mix_n_and_stft(audios)

        ret_dict = {'mag_mix': mag_mix, 'frames_features': frames_features, 'mags': mags, 'classes': class_list, 'motions': motions}
        if self.split != 'train':
            ret_dict['audios'] = audios
            ret_dict['phase_mix'] = phase_mix
            ret_dict['infos'] = infos
            ret_dict['frames'] = frames

        return ret_dict

