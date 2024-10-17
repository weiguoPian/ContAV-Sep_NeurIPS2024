import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
import clip
from PIL import Image
import numpy as np



device = 'cuda'

model, preprocess = clip.load("ViT-B/32", device=device)

train_category_vids_dict_path = '../data/train_category_vids_dict.npy'
val_category_vids_dict_path = '../data/val_category_vids_dict.npy'
test_category_vids_dict_path = '../data/test_category_vids_dict.npy'

train_category_vids_dict = np.load(train_category_vids_dict_path, allow_pickle=True).item()
val_category_vids_dict = np.load(val_category_vids_dict_path, allow_pickle=True).item()
test_category_vids_dict = np.load(test_category_vids_dict_path, allow_pickle=True).item()

train_vid_objects_features_dict = {}
val_vid_objects_features_dict = {}
test_vid_objects_features_dict = {}

# Your data root
data_root = ''
frames_root = '{}/objects'.format(data_root)

category_list = os.listdir(frames_root)
for category in category_list:
    print(category)
    category_path = os.path.join(frames_root, category)
    train_category_vid_list = train_category_vids_dict[category]
    val_category_vid_list = val_category_vids_dict[category]
    test_category_vid_list = test_category_vids_dict[category]

    print('train split...')
    for vid in train_category_vid_list:
        vid_features = []
        vid_dir = os.path.join(category_path, vid)
        num_frames = len(os.listdir(vid_dir))
        for i in range(num_frames):
            single_frame_path = os.path.join(vid_dir, str(i+1).zfill(6)+'.jpg')
            image = preprocess(Image.open(single_frame_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image).squeeze(0).cpu().numpy()
            vid_features.append(image_features)
        vid_features = np.array(vid_features)
        train_vid_objects_features_dict[vid] = vid_features
    
    print('val split...')
    for vid in val_category_vid_list:
        vid_features = []
        vid_dir = os.path.join(category_path, vid)
        num_frames = len(os.listdir(vid_dir))
        for i in range(num_frames):
            single_frame_path = os.path.join(vid_dir, str(i+1).zfill(6)+'.jpg')
            image = preprocess(Image.open(single_frame_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image).squeeze(0).cpu().numpy()
            vid_features.append(image_features)
        vid_features = np.array(vid_features)
        val_vid_objects_features_dict[vid] = vid_features

    print('test split...')
    for vid in test_category_vid_list:
        vid_features = []
        vid_dir = os.path.join(category_path, vid)
        num_frames = len(os.listdir(vid_dir))
        for i in range(num_frames):
            single_frame_path = os.path.join(vid_dir, str(i+1).zfill(6)+'.jpg')
            image = preprocess(Image.open(single_frame_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image).squeeze(0).cpu().numpy()
            vid_features.append(image_features)
        vid_features = np.array(vid_features)
        test_vid_objects_features_dict[vid] = vid_features

np.save('{}/objects_features/train_vid_objects_features_dict.npy'.format(data_root), train_vid_objects_features_dict)
np.save('{}/objects_features/val_vid_objects_features_dict.npy'.format(data_root), val_vid_objects_features_dict)
np.save('{}/objects_features/test_vid_objects_features_dict.npy'.format(data_root), test_vid_objects_features_dict)

print('Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

