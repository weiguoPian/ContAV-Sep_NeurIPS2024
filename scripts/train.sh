#!/bin/bash

OPTS=""
OPTS+="--id MUSIC21 "

# Models
OPTS+="--arch_sound unet7 "
OPTS+="--arch_frame resnet18dilated "
OPTS+="--img_pool maxpool "
OPTS+="--num_channels 32 "

# binary mask, BCE loss, weighted loss
OPTS+="--binary_mask 0 "
OPTS+="--loss l1 "
OPTS+="--weighted_loss 1 "
# logscale in frequency
OPTS+="--num_mix 2 "
OPTS+="--log_freq 1 "

#maskformer decoder parameters
OPTS+="--in_channels 256 " #"channels of the input features"
OPTS+="--MASK_FORMER_HIDDEN_DIM 256 " #"hidden_dim"
OPTS+="--MASK_FORMER_NUM_OBJECT_QUERIES 22 " #"num_queries"
OPTS+="--MASK_FORMER_NHEADS 8 " #"nheads"
OPTS+="--MASK_FORMER_DROPOUT 0 " #dropout
OPTS+="--MASK_FORMER_DIM_FEEDFORWARD 1024 " #"dim_feedforward"
OPTS+="--MASK_FORMER_ENC_LAYERS 1 "
OPTS+="--MASK_FORMER_DEC_LAYERS 4 " #"dec_layers"
OPTS+="--SEM_SEG_HEAD_MASK_DIM 32 " #"mask_dim"
OPTS+="--lr_maskformer 0.0001 "
OPTS+="--weight_decay_maskformer 0.0001 "

OPTS+="--lr_drop_maskformer 80 "

# frames-related
OPTS+="--num_frames 3 "
OPTS+="--stride_frames 1 "
OPTS+="--frameRate 1 "

# audio-related
OPTS+="--audLen 65535 " # 65535
OPTS+="--audRate 11025 " #11025

# OPTS+="--audLen 65535 "
# OPTS+="--audRate 11000 "
# OPTS+="--stft_hop 256 "

# learning params
OPTS+="--num_gpus 1 "
OPTS+="--workers 16 "
OPTS+="--batch_size_per_gpu 32 "
OPTS+="--inference_batch_size 32 "
OPTS+="--lr_frame 1e-4 "
OPTS+="--lr_sound 1e-4 "
OPTS+="--num_epoch 100 "
OPTS+="--lr_steps 30 50 "

# display, viz
OPTS+="--disp_iter 20 "
OPTS+="--num_vis 40 "
OPTS+="--num_val 256 "

OPTS+="--class_num_per_step 5 "
OPTS+="--num_classes 20 "
OPTS+="--mode train "

OPTS+="--fp16 "

OPTS+="--final_mask_distil "
OPTS+="--lam_mask_distl 0.3 "

OPTS+="--cross_modal_contra "
OPTS+="--lam_ins_contra 0.1 "
OPTS+="--lam_cls_contra 0.3 "

OPTS+="--exemplar_num_per_class 1 "


OMP_NUM_THREADS=20 CUDA_VISIBLE_DEVICES=2 nohup python -u main.py $OPTS > nohup.log 2>&1 &



