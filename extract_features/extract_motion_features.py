import os
import sys

import torch
from transformers import VideoMAEImageProcessor, VideoMAEModel

import numpy as np

from PIL import Image
import argparse
import h5py
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

ImageProcessor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

model = model.cuda()

model.eval()

# Your data root
data_root = ''

input_frames_path = "{}/frames".format(data_root)


motion_features_dict = {}

categories = os.listdir(input_frames_path)
for c in categories:
    print(c)
    c_path = os.path.join(input_frames_path, c)
    id_list = os.listdir(c_path)
    for vid in id_list:
        vid_path = os.path.join(c_path, vid)
        frames_num = len(os.listdir(vid_path))

        if frames_num > 16:
            sampling_frame_num = 16
        else:
            sampling_frame_num = frames_num

        visual_frames = []
        visual_frames.append(Image.open(os.path.join(vid_path, str(1).zfill(6) + '.jpg')))

        interval = frames_num / (sampling_frame_num - 1)

        for i in range(sampling_frame_num - 2):
            idx = 1 + int((i + 1) * interval + 0.5)
            visual_frames.append(Image.open(os.path.join(vid_path, str(idx).zfill(6) + '.jpg')))
        visual_frames.append(Image.open(os.path.join(vid_path, str(frames_num).zfill(6) + '.jpg')))

        inputs = ImageProcessor(visual_frames, return_tensors='pt')
        inputs = inputs['pixel_values']
        if inputs.shape[1] < 16:
            padding = torch.zeros((1, 16-inputs.shape[1], 3, 224, 224))
            inputs = torch.cat((inputs, padding), dim=1)

        inputs = inputs.cuda()

        with torch.no_grad():
            feature = model(inputs).last_hidden_state
        feature = feature.squeeze(dim=0).view(8, -1, 768).transpose(1, 2)
        feature = feature.view(8, 768, 14, 14)
        feature = F.adaptive_max_pool2d(feature, 1)
        feature = feature.view(8, 768).transpose(0, 1).detach().cpu().numpy()

        motion_features_dict[vid] = feature

np.save('{}/motion_features/motion_features_dict.npy'.format(data_root), motion_features_dict)

print('Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

