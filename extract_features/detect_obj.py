import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import sys
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Detic libraries
sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test

import cv2

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from detic.modeling.text.text_encoder import build_text_encoder
def get_clip_embeddings(vocabulary, prompt='a '):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb

# Your data root
data_root = ''
frames_root = '{}/frames'.format(data_root)
object_root = '{}/objects'.format(data_root)

threshold = 0.1

# Build the detector and download our pretrained weights
cfg = get_cfg()
add_centernet_config(cfg)
add_detic_config(cfg)
cfg.merge_from_file("configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set threshold for this model
cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True # For better visualization purpose. Set to False for all classes.
# cfg.MODEL.DEVICE='cpu' # uncomment this to use cpu-only mode.
# cfg.MODEL.DEVICE='gpu'
predictor = DefaultPredictor(cfg)


class_list = os.listdir(frames_root)

# i = 0
for class_name in class_list:
    print(class_name)

    print('====================================')
    print('set class...')

    thing_classes = [class_name]
    classifier = get_clip_embeddings(thing_classes)
    num_classes = len(thing_classes)
    reset_cls_test(predictor.model, classifier, num_classes)
    # Reset visualization threshold
    # output_score_threshold = threshold
    for cascade_stages in range(len(predictor.model.roi_heads.box_predictor)):
        predictor.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = threshold
    print('====================================')

    class_root = os.path.join(frames_root, class_name)
    v_name_list = os.listdir(class_root)
    for v_name in v_name_list:
        v_frames_dir = os.path.join(class_root, v_name)
        frames_list = os.listdir(v_frames_dir)
        for frame_id in frames_list:
            single_frame_path = os.path.join(v_frames_dir, frame_id)
            im = cv2.imread(single_frame_path)
            outputs = predictor(im)
            # print(outputs["instances"].pred_boxes)
            obj_pos_list = outputs["instances"].pred_boxes.tensor.detach().cpu().numpy()
            if len(obj_pos_list) > 0:
                obj_pos = obj_pos_list[0]
                obj = im[int(obj_pos[1]):int(obj_pos[3]), int(obj_pos[0]):int(obj_pos[2]), :]
            else:
                thing_classes = ['person is playing {}'.format(class_name)]
                classifier = get_clip_embeddings(thing_classes)
                num_classes = len(thing_classes)
                reset_cls_test(predictor.model, classifier, num_classes)
                for cascade_stages in range(len(predictor.model.roi_heads.box_predictor)):
                    predictor.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = threshold
                obj_pos_list = outputs["instances"].pred_boxes.tensor.detach().cpu().numpy()
                if len(obj_pos_list) > 0:
                    obj_pos = obj_pos_list[0]
                    obj = im[int(obj_pos[1]):int(obj_pos[3]), int(obj_pos[0]):int(obj_pos[2]), :]
                else:
                    thing_classes = ['person']
                    classifier = get_clip_embeddings(thing_classes)
                    num_classes = len(thing_classes)
                    reset_cls_test(predictor.model, classifier, num_classes)
                    for cascade_stages in range(len(predictor.model.roi_heads.box_predictor)):
                        predictor.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = threshold
                    obj_pos_list = outputs["instances"].pred_boxes.tensor.detach().cpu().numpy()
                    if len(obj_pos_list) > 0:
                        obj_pos = obj_pos_list[0]
                        obj = im[int(obj_pos[1]):int(obj_pos[3]), int(obj_pos[0]):int(obj_pos[2]), :]
                    else:
                        obj = im
                
                thing_classes = [class_name]
                classifier = get_clip_embeddings(thing_classes)
                num_classes = len(thing_classes)

                reset_cls_test(predictor.model, classifier, num_classes)
                for cascade_stages in range(len(predictor.model.roi_heads.box_predictor)):
                    predictor.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = threshold
                
            
            objs_save_dir = os.path.join(object_root, class_name, v_name)
            if not os.path.exists(objs_save_dir):
                os.makedirs(objs_save_dir)
            single_obj_save_path = os.path.join(objs_save_dir, frame_id)
            cv2.imwrite(single_obj_save_path, obj)


