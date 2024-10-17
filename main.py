# System libs
import os
import random
import time

# Numerical libs
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import scipy.io.wavfile as wavfile
import cv2
from PIL import Image
from mir_eval.separation import bss_eval_sources

from torch.utils.data.dataloader import default_collate
import copy

# Our libs
from arguments import ArgParser
from dataset import MUSICMix21IncrementalOursDataset
from models import ModelBuilder, activate
from utils import AverageMeter, \
    recover_rgb, magnitude2heatmap, \
    istft_reconstruction, warpgrid, \
    combine_video_audio, save_video, makedirs
from viz import plot_loss_metrics, HTMLVisualizer
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

from tqdm import tqdm
from tqdm.contrib import tzip

import warnings
warnings.filterwarnings("ignore")

class LBSign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()

    def forward(self, m):

        return torch.mean(1.0/(1e-6 + torch.abs(m-0.5)))

def tf_data(x, B, s1, s2):
    return torch.softmax(x, dim=-1)[:, 0].view(B, 1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, s1, s2)

# Network wrapper, defines forward pass
class NetWrapper(torch.nn.Module):
    def __init__(self, step, maskformer, nets, crit):
        super(NetWrapper, self).__init__()
        self.net_sound = nets
        self.crit = crit
        self.cts = nn.CrossEntropyLoss()
        self.bn = LBSign.apply
        self.cts_bn = BinaryLoss()
        self.net_maskformer = maskformer
        self.step = step

        self.old_net_sound, self.old_net_maskformer = None, None

    def _init_old_model_(self):
        self.old_net_sound = copy.deepcopy(self.net_sound)
        self.old_net_maskformer = copy.deepcopy(self.net_maskformer)

        self.old_net_sound.requires_grad_ = False
        self.old_net_maskformer.requires_grad_ = False


    def load_weights(self, args, step):
        print('Loading weights...')

        best_sound_weights = torch.load('{}/sound_step_{}_best.pth'.format(args.ckpt, step), map_location='cpu')
        best_maskformer_weights = torch.load('{}/maskformer_step_{}_best.pth'.format(args.ckpt, step), map_location='cpu')
        
        self.net_sound.load_state_dict(best_sound_weights)
        self.net_maskformer.load_state_dict(best_maskformer_weights)

    def model_incremental(self, num_classes):
        print('================ Query Embedding Incremental... =====================')
        self.net_maskformer.incremental_query(num_classes)

    def forward(self, batch_data, args, mode='eval', step_class_sim_matrix=None):
        if mode != 'train' and mode != 'eval':
            raise ValueError('mode must be \'train\' or \'eval\'.')

        mag_mix = batch_data['mag_mix']
        mags = batch_data['mags']
        feat_frames = batch_data['frames_features']
        classes = batch_data['classes']

        motions = batch_data['motions']
        motion_masks = batch_data['motion_masks']
        
        mag_mix = mag_mix + 1e-10

        N = args.num_mix
        B = mag_mix.size(0)
        T = mag_mix.size(3)

        if self.step > 0:
            memory_data_indexes = []
            new_data_indexes = []
            for i in range(N):
                old_indexes = []
                new_indexes = []
                for j in range(B):
                    if classes[i][j].item() < self.step * args.class_num_per_step:
                        old_indexes.append(j)
                    else:
                        new_indexes.append(j)
                old_indexes = torch.Tensor(old_indexes).long()
                new_indexes = torch.Tensor(new_indexes).long()
                memory_data_indexes.append(old_indexes)
                new_data_indexes.append(new_indexes)
            if len(torch.concat(memory_data_indexes)) == 0:
                include_old_data = False
            else:
                include_old_data = True

        mag_mix = mag_mix.to(args.device)
        for i in range(N):
            mags[i] = mags[i].to(args.device)
            feat_frames[i] = feat_frames[i].to(args.device)
            classes[i] = classes[i].to(args.device)
            motions[i] = motions[i].to(args.device)
            motion_masks[i] = motion_masks[i].to(args.device)

        # 0.0 warp the spectrogram
        if args.log_freq:
            grid_warp = torch.from_numpy(
                warpgrid(B, 256, T, warp=True)).to(args.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp)
            for n in range(N):
                mags[n] = F.grid_sample(mags[n], grid_warp)

        # 0.1 calculate loss weighting coefficient: magnitude of input mixture
        if args.weighted_loss:
            weight = torch.log1p(mag_mix)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = torch.ones_like(mag_mix)

        # 0.2 ground truth masks are computed after warpping!
        gt_masks = [None for n in range(N)]
        for n in range(N):
            if args.binary_mask:
                gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()
            else:
                gt_masks[n] = mags[n] / mag_mix
                # clamp to avoid large numbers in ratio masks
                gt_masks[n].clamp_(0., 5.)

        # LOG magnitude
        log_mag_mix = torch.log1p(mag_mix).detach()
        
        feat_sound, feat_latent = self.net_sound(log_mag_mix)
        feat_sound = activate(feat_sound, args.sound_activation)
        feat_latent = activate(feat_latent, args.sound_activation)

        if mode == 'train' and self.step > 0:
            old_feat_sound, old_feat_latent = self.old_net_sound(log_mag_mix)
            old_feat_sound = old_feat_sound.detach()
            old_feat_latent = old_feat_latent.detach()
            old_feat_sound = activate(old_feat_sound, args.sound_activation)
            old_feat_latent = activate(old_feat_latent, args.sound_activation)

        pred_masks = [None for n in range(N)]
        inter_motion_feat = [None for n in range(N)]
        inter_obj_feat = [None for n in range(N)]
        inter_sound_latent_feat = [None for n in range(N)]
        sep_audio_feat = [None for n in range(N)]

        if mode == 'train' and self.step > 0:
            old_data_pred_masks = [None for n in range(N)]
        for n in range(N):
            pred_masks[n], sep_audio_feat[n], inter_sound_latent_feat[n], inter_motion_feat[n], inter_obj_feat[n] = self.net_maskformer(feat_latent, feat_sound, classes[n], feat_frames[n], motions[n], motion_masks[n], return_sound_motion_obj_feat=True)
            pred_masks[n] = activate(pred_masks[n], args.output_activation)
        
        # 5. loss
        loss_sep = self.crit(pred_masks[0], gt_masks[0], weight).reshape(1)
        for i in range(1, N):
            loss_sep += self.crit(pred_masks[i], gt_masks[i], weight).reshape(1)
        
        err = loss_sep / N

        if mode == 'train' and args.cross_modal_contra:
            if step == 0 or not include_old_data:
                cat_sep_audio_feat = torch.cat(sep_audio_feat)
                cat_inter_motion_feat = torch.cat(inter_motion_feat)
                cat_inter_obj_feat = torch.cat(inter_obj_feat)

                all_data_classes = torch.concat(classes)

                instance_equal_matrix = torch.eq(cat_inter_motion_feat[:, None], cat_inter_motion_feat).all(dim=2).float().detach()

                class_equal_matrix = all_data_classes.unsqueeze(0)
                class_equal_matrix = class_equal_matrix.repeat(class_equal_matrix.shape[1], 1)
                class_equal_matrix = class_equal_matrix == all_data_classes.unsqueeze(-1)
                class_equal_matrix = class_equal_matrix.float().detach()

                if args.cross_modal_contra:
                    instance_contra_loss = self.cross_modal_contrastive_loss(
                        F.normalize(cat_sep_audio_feat), 
                        F.normalize(cat_inter_motion_feat), 
                        F.normalize(cat_inter_obj_feat), 
                        equal_matrix=instance_equal_matrix,
                        temperature=0.05)

                    class_contra_loss = self.cross_modal_contrastive_loss(
                        F.normalize(cat_sep_audio_feat), 
                        F.normalize(cat_inter_motion_feat), 
                        F.normalize(cat_inter_obj_feat), 
                        equal_matrix=class_equal_matrix,
                        temperature=0.05)

                    err += args.lam_ins_contra * instance_contra_loss
                    err += args.lam_cls_contra * class_contra_loss

        if mode == 'train' and self.step > 0 and include_old_data:
            old_data_classes = []
            old_data_feat_frames = []
            old_data_motions = []
            old_data_motion_masks = []

            new_data_classes = []

            old_data_new_model_pred_masks = []
            old_data_new_model_sep_audio_feat = []
            old_data_new_model_inter_sound_latent_feat = []
            old_data_new_model_inter_motion_feat = []
            old_data_new_model_inter_obj_feat = []

            new_data_pred_masks = []
            new_data_sep_audio_feat = []
            new_data_inter_sound_latent_feat = []
            new_data_inter_motion_feat = []
            new_data_inter_obj_feat = []

            for n in range(N):
                old_data_classes.append(classes[n][memory_data_indexes[n]])
                old_data_feat_frames.append(feat_frames[n][memory_data_indexes[n]])
                old_data_motions.append(motions[n][memory_data_indexes[n]])
                old_data_motion_masks.append(motion_masks[n][memory_data_indexes[n]])

                old_data_new_model_pred_masks.append(pred_masks[n][memory_data_indexes[n]])
                old_data_new_model_sep_audio_feat.append(sep_audio_feat[n][memory_data_indexes[n]])
                old_data_new_model_inter_sound_latent_feat.append(inter_sound_latent_feat[n][memory_data_indexes[n]])
                old_data_new_model_inter_motion_feat.append(inter_motion_feat[n][memory_data_indexes[n]])
                old_data_new_model_inter_obj_feat.append(inter_obj_feat[n][memory_data_indexes[n]])

                new_data_classes.append(classes[n][new_data_indexes[n]])

                new_data_pred_masks.append(pred_masks[n][new_data_indexes[n]])
                new_data_sep_audio_feat.append(sep_audio_feat[n][new_data_indexes[n]])
                new_data_inter_sound_latent_feat.append(inter_sound_latent_feat[n][new_data_indexes[n]])
                new_data_inter_motion_feat.append(inter_motion_feat[n][new_data_indexes[n]])
                new_data_inter_obj_feat.append(inter_obj_feat[n][new_data_indexes[n]])
            
            old_data_feat_latent = old_feat_latent[torch.concat(memory_data_indexes)]
            old_data_feat_sound = old_feat_sound[torch.concat(memory_data_indexes)]

            old_data_weight = weight[torch.concat(memory_data_indexes)]

            old_data_new_model_pred_masks = torch.concat(old_data_new_model_pred_masks, dim=0)
            old_data_new_model_sep_audio_feat = torch.concat(old_data_new_model_sep_audio_feat, dim=0)
            old_data_new_model_inter_sound_latent_feat = torch.concat(old_data_new_model_inter_sound_latent_feat, dim=0)
            old_data_new_model_inter_motion_feat = torch.concat(old_data_new_model_inter_motion_feat, dim=0)
            old_data_new_model_inter_obj_feat = torch.concat(old_data_new_model_inter_obj_feat, dim=0)

            new_data_pred_masks = torch.concat(new_data_pred_masks, dim=0)
            new_data_sep_audio_feat = torch.concat(new_data_sep_audio_feat, dim=0)
            new_data_inter_sound_latent_feat = torch.concat(new_data_inter_sound_latent_feat, dim=0)
            new_data_inter_motion_feat = torch.concat(new_data_inter_motion_feat, dim=0)
            new_data_inter_obj_feat = torch.concat(new_data_inter_obj_feat, dim=0)
            
            old_data_classes = torch.concat(old_data_classes, dim=0)
            old_data_feat_frames = torch.concat(old_data_feat_frames, dim=0)
            old_data_motions = torch.concat(old_data_motions, dim=0)
            old_data_motion_masks = torch.concat(old_data_motion_masks, dim=0)

            new_data_classes = torch.concat(new_data_classes, dim=0)
            
            old_data_pred_masks, \
            old_data_sep_audio_feat, \
            old_data_inter_sound_latent_feat, \
            old_data_inter_motion_feat, \
            old_data_inter_obj_feat = self.old_net_maskformer(
                old_data_feat_latent, 
                old_data_feat_sound, 
                old_data_classes, 
                old_data_feat_frames, 
                old_data_motions, 
                old_data_motion_masks, 
                return_sound_motion_obj_feat=True)
            
            old_data_pred_masks = activate(old_data_pred_masks, args.output_activation).detach()
            old_data_sep_audio_feat = old_data_sep_audio_feat.detach()
            old_data_inter_sound_latent_feat = old_data_inter_sound_latent_feat.detach()
            old_data_inter_motion_feat = old_data_inter_motion_feat.detach()
            old_data_inter_obj_feat = old_data_inter_obj_feat.detach()

            if args.cross_modal_contra:
                new_data_num = new_data_pred_masks.shape[0]
                old_data_num = old_data_pred_masks.shape[0]
                assert old_data_num == old_data_new_model_pred_masks.shape[0]

                instance_old_equal_matrix = torch.eq(old_data_inter_motion_feat[:, None], old_data_inter_motion_feat).all(dim=2).float().detach()
                instance_old_equal_matrix = instance_old_equal_matrix.repeat(2, 2)

                instance_new_equal_matrix = torch.eq(new_data_inter_motion_feat[:, None], new_data_inter_motion_feat).all(dim=2).float().detach()

                instance_equal_matrix = torch.zeros(
                    instance_new_equal_matrix.shape[0]+instance_old_equal_matrix.shape[0], instance_new_equal_matrix.shape[0]+instance_old_equal_matrix.shape[0]).to(instance_old_equal_matrix.device)
                instance_equal_matrix[:instance_new_equal_matrix.shape[0], :instance_new_equal_matrix.shape[0]] = instance_new_equal_matrix
                instance_equal_matrix[instance_new_equal_matrix.shape[0]:, instance_new_equal_matrix.shape[0]:] = instance_old_equal_matrix
                instance_equal_matrix = instance_equal_matrix.detach()

                all_data_classes = torch.concat((new_data_classes, old_data_classes, old_data_classes), dim=0)

                class_equal_matrix = all_data_classes.unsqueeze(0)
                class_equal_matrix = class_equal_matrix.repeat(class_equal_matrix.shape[1], 1)
                class_equal_matrix = class_equal_matrix == all_data_classes.unsqueeze(-1)
                class_equal_matrix = class_equal_matrix.float().detach()

                cat_sep_audio_feat = torch.cat((new_data_sep_audio_feat, old_data_new_model_sep_audio_feat, old_data_sep_audio_feat), dim=0)
                cat_inter_motion_feat = torch.cat((new_data_inter_motion_feat, old_data_new_model_inter_motion_feat, old_data_inter_motion_feat), dim=0)
                cat_inter_obj_feat = torch.cat((new_data_inter_obj_feat, old_data_new_model_inter_obj_feat, old_data_inter_obj_feat), dim=0)

                instance_contra_loss = self.cross_modal_contrastive_loss(
                    F.normalize(cat_sep_audio_feat), 
                    F.normalize(cat_inter_motion_feat), 
                    F.normalize(cat_inter_obj_feat), 
                    temperature=0.05,
                    equal_matrix=instance_equal_matrix)
                
                class_contra_loss = self.cross_modal_contrastive_loss(
                    F.normalize(cat_sep_audio_feat), 
                    F.normalize(cat_inter_motion_feat), 
                    F.normalize(cat_inter_obj_feat), 
                    temperature=0.05,
                    equal_matrix=class_equal_matrix)

                err += args.lam_ins_contra * instance_contra_loss
                err += args.lam_cls_contra * class_contra_loss

            if args.final_mask_distil:
                final_mask_distil_distl_loss = self.crit(old_data_new_model_pred_masks, old_data_pred_masks, old_data_weight).reshape(1)
                err += args.lam_mask_distl * final_mask_distil_distl_loss
        
        return err, \
            {'pred_masks': pred_masks, 'gt_masks': gt_masks,
             'mag_mix': mag_mix, 'mags': mags, 'weight': weight}
    
    def CE_loss(self, num_classes, logits, label):
        targets = F.one_hot(label, num_classes=num_classes)
        loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=-1) * targets, dim=1))

        return loss
    
    def CE_loss_(self, score, label_matrix):
        loss = -torch.mean(
            (torch.sum(F.log_softmax(score, dim=-1) * label_matrix, dim=-1)) / torch.sum(label_matrix, dim=-1))
        return loss

    def cross_modal_contrastive_loss(self, sound_feat, motion_feat, obj_feat, equal_matrix, temperature=0.05):
        
        score_sound_motion = torch.mm(sound_feat, motion_feat.transpose(0, 1)) / temperature
        score_sound_obj = torch.mm(sound_feat, obj_feat.transpose(0, 1)) / temperature
        score_motion_obj = torch.mm(motion_feat, obj_feat.transpose(0, 1)) / temperature

        loss_sound_motion = (self.CE_loss_(score_sound_motion, equal_matrix) + self.CE_loss_(score_sound_motion.transpose(0, 1), equal_matrix)) * 0.5
        loss_sound_obj = (self.CE_loss_(score_sound_obj, equal_matrix) + self.CE_loss_(score_sound_obj.transpose(0, 1), equal_matrix)) * 0.5
        loss_motion_obj = (self.CE_loss_(score_motion_obj, equal_matrix) + self.CE_loss_(score_motion_obj.transpose(0, 1), equal_matrix)) * 0.5

        loss = (loss_sound_motion + loss_sound_obj + loss_motion_obj) / 3.

        return loss


# Calculate metrics
def calc_metrics(batch_data, outputs, args):
    # meters
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    sir_mix_list = []
    sdr_list = []
    sir_list = []
    sar_list = []

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
            pred_masks_linear[n] = (pred_masks_linear[n] > args.mask_thres).astype(np.float32)

    # loop over each sample
    for j in range(B):
        # save mixture
        mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)

        # save each component
        preds_wav = [None for n in range(N)]
        for n in range(N):
            # Predicted audio recovery
            pred_mag = mag_mix[j, 0] * pred_masks_linear[n][j, 0]
            preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)

        # separation performance computes
        L = preds_wav[0].shape[0]
        gts_wav = [None for n in range(N)]
        valid = True
        for n in range(N):
            gts_wav[n] = audios[n][j, 0:L].numpy()
            valid *= np.sum(np.abs(gts_wav[n])) > 1e-5
            valid *= np.sum(np.abs(preds_wav[n])) > 1e-5
        if valid:
            sdr, sir, sar, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray(preds_wav),
                False)
            sdr_mix, _, _, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray([mix_wav[0:L] for n in range(N)]),
                False)
            sir_mix_list.append(sdr_mix.mean())
            sdr_list.append(sdr.mean())
            sir_list.append(sir.mean())
            sar_list.append(sar.mean())

    return sir_mix_list, sdr_list, sir_list, sar_list


# Visualize predictions
def output_visuals(vis_rows, batch_data, outputs, args, mode='val'):
    if mode == 'val':
        output_root = args.vis_val
    else:
        output_root = args.vis_test

    # fetch data and predictions
    mag_mix = batch_data['mag_mix']
    phase_mix = batch_data['phase_mix']
    frames = batch_data['frames']
    infos = batch_data['infos']

    pred_masks_ = outputs['pred_masks']
    gt_masks_ = outputs['gt_masks']
    mag_mix_ = outputs['mag_mix']
    weight_ = outputs['weight']

    # unwarp log scale
    N = args.num_mix
    B = mag_mix.size(0)
    pred_masks_linear = [None for n in range(N)]
    gt_masks_linear = [None for n in range(N)]
    for n in range(N):
        if args.log_freq:
            grid_unwarp = torch.from_numpy(
                warpgrid(B, args.stft_frame//2+1, gt_masks_[0].size(3), warp=False)).to(args.device)
            pred_masks_linear[n] = F.grid_sample(pred_masks_[n], grid_unwarp)
            gt_masks_linear[n] = F.grid_sample(gt_masks_[n], grid_unwarp)
        else:
            pred_masks_linear[n] = pred_masks_[n]
            gt_masks_linear[n] = gt_masks_[n]

    # convert into numpy
    mag_mix = mag_mix.numpy()
    mag_mix_ = mag_mix_.detach().cpu().numpy()
    phase_mix = phase_mix.numpy()
    weight_ = weight_.detach().cpu().numpy()

    for n in range(N):
        pred_masks_[n] = pred_masks_[n].detach().cpu().numpy()
        pred_masks_linear[n] = pred_masks_linear[n].detach().cpu().numpy()
        gt_masks_[n] = gt_masks_[n].detach().cpu().numpy()
        gt_masks_linear[n] = gt_masks_linear[n].detach().cpu().numpy()

        frames[n] = frames[n].detach().cpu()

        # threshold if binary mask
        if args.binary_mask:
            pred_masks_[n] = (pred_masks_[n] > args.mask_thres).astype(np.float32)
            pred_masks_linear[n] = (pred_masks_linear[n] > args.mask_thres).astype(np.float32)

    # loop over each sample
    for j in range(B):
        row_elements = []

        # video names
        prefix = []
        for n in range(N):
            prefix.append('-'.join(infos[n][0][j].split('/')[-2:]).split('.')[0])
        prefix = '+'.join(prefix)
        makedirs(os.path.join(output_root, prefix))

        # save mixture
        mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)
        mix_amp = magnitude2heatmap(mag_mix_[j, 0])
        weight = magnitude2heatmap(weight_[j, 0], log=False, scale=100.)
        filename_mixwav = os.path.join(prefix, 'mix.wav')
        filename_mixmag = os.path.join(prefix, 'mix.jpg')
        filename_weight = os.path.join(prefix, 'weight.jpg')
        cv2.imwrite(os.path.join(output_root, filename_mixmag), mix_amp[::-1, :, :])
        cv2.imwrite(os.path.join(output_root, filename_weight), weight[::-1, :])
        wavfile.write(os.path.join(output_root, filename_mixwav), args.audRate, mix_wav)
        row_elements += [{'text': prefix}, {'image': filename_mixmag, 'audio': filename_mixwav}]

        # save each component
        preds_wav = [None for n in range(N)]
        for n in range(N):
            # GT and predicted audio recovery
            gt_mag = mag_mix[j, 0] * gt_masks_linear[n][j, 0]
            gt_wav = istft_reconstruction(gt_mag, phase_mix[j, 0], hop_length=args.stft_hop)
            pred_mag = mag_mix[j, 0] * pred_masks_linear[n][j, 0]
            preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)

            # output masks
            filename_gtmask = os.path.join(prefix, 'gtmask{}.jpg'.format(n+1))
            filename_predmask = os.path.join(prefix, 'predmask{}.jpg'.format(n+1))
            gt_mask = (np.clip(gt_masks_[n][j, 0], 0, 1) * 255).astype(np.uint8)
            pred_mask = (np.clip(pred_masks_[n][j, 0], 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_root, filename_gtmask), gt_mask[::-1, :])
            cv2.imwrite(os.path.join(output_root, filename_predmask), pred_mask[::-1, :])

            # ouput spectrogram (log of magnitude, show colormap)
            filename_gtmag = os.path.join(prefix, 'gtamp{}.jpg'.format(n+1))
            filename_predmag = os.path.join(prefix, 'predamp{}.jpg'.format(n+1))
            gt_mag = magnitude2heatmap(gt_mag)
            pred_mag = magnitude2heatmap(pred_mag)
            cv2.imwrite(os.path.join(output_root, filename_gtmag), gt_mag[::-1, :, :])
            cv2.imwrite(os.path.join(output_root, filename_predmag), pred_mag[::-1, :, :])

            # output audio
            filename_gtwav = os.path.join(prefix, 'gt{}.wav'.format(n+1))
            filename_predwav = os.path.join(prefix, 'pred{}.wav'.format(n+1))
            wavfile.write(os.path.join(output_root, filename_gtwav), args.audRate, gt_wav)
            wavfile.write(os.path.join(output_root, filename_predwav), args.audRate, preds_wav[n])

            # output video
            frames_tensor = [recover_rgb(frames[n][j, :, t]) for t in range(args.num_frames)]
            frames_tensor = np.asarray(frames_tensor)
            path_video = os.path.join(output_root, prefix, 'video{}.mp4'.format(n+1))
            save_video(path_video, frames_tensor, fps=args.frameRate/args.stride_frames)

            # combine gt video and audio
            filename_av = os.path.join(prefix, 'av{}.mp4'.format(n+1))
            combine_video_audio(
                path_video,
                os.path.join(output_root, filename_gtwav),
                os.path.join(output_root, filename_av))

            row_elements += [
                {'video': filename_av},
                {'image': filename_predmag, 'audio': filename_predwav},
                {'image': filename_gtmag, 'audio': filename_gtwav},
                {'image': filename_predmask},
                {'image': filename_gtmask}]

        row_elements += [{'image': filename_weight}]
        vis_rows.append(row_elements)

def visualization(netWrapper, loader, args, mode='val'):
    if mode == 'val':
        vis_root = args.vis_val
    else:
        vis_root = args.vis_test
    makedirs(vis_root, remove=True)
    netWrapper.eval()

    visualizer = HTMLVisualizer(os.path.join(vis_root, 'index.html'))
    header = ['Filename', 'Input Mixed Audio']
    for n in range(1, args.num_mix+1):
        header += ['Video {:d}'.format(n),
                   'Predicted Audio {:d}'.format(n),
                   'GroundTruth Audio {}'.format(n),
                   'Predicted Mask {}'.format(n),
                   'GroundTruth Mask {}'.format(n)]
    header += ['Loss weighting']
    visualizer.add_header(header)
    vis_rows = []
    
    for i, batch_data in enumerate(loader):
        _, outputs = netWrapper.forward(batch_data, args)
        if len(vis_rows) < args.num_vis:
            output_visuals(vis_rows, batch_data, outputs, args, mode=mode)
        else:
            break

    print('Plotting html for visualization...')
    visualizer.add_rows(vis_rows)
    visualizer.write_html()


def evaluate(netWrapper, loader, history, epoch, args, step, mode='val'):
    if mode != 'val' and mode != 'test':
            raise ValueError('mode must be \'val\' or \'test\'.')
    if mode == 'val':
        print('Evaluating at {} epochs...'.format(epoch))
    torch.set_grad_enabled(False)

    # switch to eval mode
    netWrapper.eval()

    # initialize meters
    loss_meter = AverageMeter()
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    all_sdr_mix_list, all_sdr_list, all_sir_list, all_sar_list = [], [], [], []

    for batch_data in tqdm(loader):
        # forward pass
        err, outputs = netWrapper.forward(batch_data, args)
        err = err.mean()

        loss_meter.update(err.item())

        # calculate metrics
        sdr_mix_list, sdr_list, sir_list, sar_list = calc_metrics(batch_data, outputs, args)

        all_sdr_mix_list += sdr_mix_list
        all_sdr_list += sdr_list
        all_sir_list += sir_list
        all_sar_list += sar_list
    
    mean_sdr_mix = np.mean(all_sdr_mix_list)
    mean_sdr = np.mean(all_sdr_list)
    mean_sir = np.mean(all_sir_list)
    mean_sar = np.mean(all_sar_list)

    if mode == 'val':
        history['val']['epoch'].append(epoch)
        history['val']['err'].append(loss_meter.average())
        history['val']['sdr'].append(mean_sdr)
        history['val']['sir'].append(mean_sir)
        history['val']['sar'].append(mean_sar)

    # Plot figure
    if mode == 'val' and epoch > 0:
        print('Plotting figures...')
        plot_loss_metrics(args.ckpt, history, step)

    return loss_meter.average(), mean_sdr_mix, mean_sdr, mean_sir, mean_sar
    # return loss_meter.average(), sdr_mix_meter.average(), sdr_meter.average(), sir_meter.average(), sar_meter.average()


# train one epoch
def train(netWrapper, loader, optimizer, history, epoch, mask_optimizer, mask_lr_scheduler, args):
    torch.set_grad_enabled(True)
    netWrapper.train()

    scaler = torch.cuda.amp.GradScaler()

    # main loop
    i = 0
    train_loss = 0.0
    for batch_data in tqdm(loader):
        # forward pass
        netWrapper.zero_grad()

        if args.fp16:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss, _ = netWrapper.forward(batch_data, args, mode='train')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.step(mask_optimizer)
            scaler.update()
        else:
            loss, _ = netWrapper.forward(batch_data, args, mode='train')
            # loss = loss.mean()
            loss.backward()
            optimizer.step()
            mask_optimizer.step()
        
        train_loss += loss.item()
        i += 1
    
    train_loss /= i
    print('Epoch: [{}], loss: {:.4f}'.format(epoch, train_loss))
    history['train']['epoch'].append(epoch)
    history['train']['err'].append(train_loss)
    mask_lr_scheduler.step()


def checkpoint(step, nets, maskformer, history, epoch, args, save_as_best=False):
    print('Saving checkpoints at {} epochs.'.format(epoch))
    net_sound = nets
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'

    torch.save(history,'{}/step_{}_history'.format(args.ckpt, step))
    torch.save(net_sound.state_dict(), '{}/sound_step_{}_{}'.format(args.ckpt, step, suffix_latest))
    torch.save(maskformer.state_dict(), '{}/maskformer_step_{}_{}'.format(args.ckpt, step, suffix_latest))

    if save_as_best:
        # args.best_sdr = cur_sdr
        print('Saving best model at Epoch {}'.format(epoch))
        torch.save(net_sound.state_dict(), '{}/sound_step_{}_{}'.format(args.ckpt, step, suffix_best))
        torch.save(maskformer.state_dict(), '{}/maskformer_step_{}_{}'.format(args.ckpt, step, suffix_best))

def create_optimizer(net_sound, args):
    param_groups = [{'params': net_sound.parameters(), 'lr': args.lr_sound}]
    return torch.optim.Adam(param_groups)

def maskformer_optimizer(model, args):
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad]}
    ]
    return torch.optim.AdamW(param_dicts, lr=args.lr_maskformer, weight_decay=args.weight_decay_maskformer)

def adjust_maskformer_learning_rate(optimizer, args):
    return torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_maskformer)

def adjust_learning_rate(optimizer, args):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1

def time_collate_fn(batch):
    new_batch = {}

    motion_1 = []
    motion_2 = []

    batch_copy = copy.deepcopy(batch)

    for raw_dict in batch:
        cur_motion1 = raw_dict['motions'][0]
        cur_motion2 = raw_dict['motions'][1]
        motion_1.append(cur_motion1)
        motion_2.append(cur_motion2)
    for copy_dict in batch_copy:
        del copy_dict['motions']

    max_t_1 = 0
    max_t_2 = 0
    for i in range(len(motion_1)):
        motion_dim, cur_t = motion_1[i].shape
        if(cur_t>max_t_1):
            max_t_1 = cur_t
    for i in range(len(motion_2)):
        motion_dim, cur_t= motion_2[i].shape
        if(cur_t>max_t_2):
            max_t_2 = cur_t
    final_motion_1 = torch.zeros((len(motion_1), motion_dim ,max_t_1)) #B, C, T
    final_motion_2 = torch.zeros((len(motion_2), motion_dim ,max_t_2))
    final_mask_1 = torch.ones((len(motion_1), max_t_1))
    final_mask_2 = torch.ones((len(motion_2), max_t_2))
    for i in range(len(motion_1)):
        cur_t_1 = motion_1[i].shape[1]
        cur_t_2 = motion_2[i].shape[1]
        final_motion_1[i,:,:cur_t_1] = motion_1[i]
        final_mask_1[i,:cur_t_1] = 0 
        final_motion_2[i,:,:cur_t_2] = motion_2[i]
        final_mask_2[i,:cur_t_2] = 0

    batch_motion_list = [final_motion_1, final_motion_2]
    batch_mask_list = [final_mask_1, final_mask_2]
    new_batch = default_collate(batch_copy)
    new_batch['motion_masks'] = batch_mask_list
    new_batch['motions'] = batch_motion_list
    
    del batch_copy
    return new_batch

def main(args, step, dataset_train, dataset_val, dataset_test):
    # Network Builders
    builder = ModelBuilder()
    net_sound = builder.build_sound(
        arch=args.arch_sound,
        fc_dim=args.num_channels,
        weights=args.weights_sound)

    if step == 0:
        num_queries = args.class_num_per_step
    else:
        num_queries = args.class_num_per_step * step

    maskformer = builder.build_maskformermotion(
        in_channels=args.in_channels,
        hidden_dim=args.MASK_FORMER_HIDDEN_DIM,
        num_queries=num_queries,
        nheads=args.MASK_FORMER_NHEADS,
        dropout=args.MASK_FORMER_DROPOUT,
        dim_feedforward=args.MASK_FORMER_DIM_FEEDFORWARD,
        enc_layers=args.MASK_FORMER_ENC_LAYERS,
        dec_layers=args.MASK_FORMER_DEC_LAYERS,
        mask_dim=args.SEM_SEG_HEAD_MASK_DIM,
        weights = args.weights_maskformer
    )

    crit = builder.build_criterion(arch=args.loss)

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers), 
        collate_fn = time_collate_fn,
        drop_last=True)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.inference_batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn = time_collate_fn,
        drop_last=False)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.inference_batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn = time_collate_fn,
        drop_last=False)

    # Wrap networks
    netWrapper = NetWrapper(step, maskformer, net_sound, crit)

    if step > 0 and not args.upper_bound:
        netWrapper.load_weights(args, step-1)
        netWrapper._init_old_model_()
        netWrapper.model_incremental((step + 1) * args.class_num_per_step)

    if torch.cuda.device_count() > 1:
        netWrapper = torch.nn.DataParallel(netWrapper)
    netWrapper.to(args.device)

    if args.mode == 'eval':
        print('Eval mode...')
        testing(args, netWrapper, loader_test, step)
        return

    # Set up optimizer
    optimizer = create_optimizer(net_sound, args)
    mask_optimizer = maskformer_optimizer(maskformer, args)
    mask_lr_scheduler = adjust_maskformer_learning_rate(mask_optimizer, args)
    
    # History of peroformance
    history = {
        'train': {'epoch': [], 'err': []},
        'val': {'epoch': [], 'err': [], 'sdr': [], 'sir': [], 'sar': []}}

    best_val_sdr = None
    best_epoch = None

    # Training loop
    for epoch in range(args.num_epoch):
        train(netWrapper, loader_train, optimizer, history, epoch, mask_optimizer, mask_lr_scheduler, args)

        # Evaluation and visualization
        if epoch % args.eval_epoch == 0:
            val_loss, val_mix_sdr, val_sdr, val_sir, val_sar = evaluate(netWrapper, loader_val, history, epoch, args, step, mode='val')
            print('[Eval Summary] Epoch: {}, Loss: {:.4f}, '
                  'SDR_mixture: {:.4f}, SDR: {:.4f}, SIR: {:.4f}, SAR: {:.4f}'
                  .format(epoch, val_loss, val_mix_sdr, val_sdr, val_sir, val_sar))
            
            if best_val_sdr is None or best_val_sdr < val_sdr:
                best_val_sdr = val_sdr
                best_epoch = epoch
                checkpoint(step, net_sound, maskformer, history, epoch, args, save_as_best=True)
                print('Visualization at Epoch {}'.format(epoch))
                ##############################################
                visualization(netWrapper, loader_val, args, mode='val')
                ##############################################
            else:
                checkpoint(step, net_sound, maskformer, history, epoch, args, save_as_best=False)

        # drop learning rate
        if epoch in args.lr_steps:
            adjust_learning_rate(optimizer, args)

    print('Training Done!')

    testing(args, netWrapper, loader_test, step)
    visualization(netWrapper, loader_test, args, mode='test')


def testing(args, netWrapper, loader, step):
    print('Starting testing...')

    if torch.cuda.device_count() > 1: 
        netWrapper.module.load_weights(args, step)
    else:
        netWrapper.load_weights(args, step)

    test_loss, test_mix_sdr, test_sdr, test_sir, test_sar = evaluate(netWrapper, loader, history=None, epoch=0, args=args, step=step, mode='test')
    print('[Testing Results] Loss: {:.4f}, SDR_mixture: {:.4f}, SDR: {:.4f}, SIR: {:.4f}, SAR: {:.4f}'
          .format(test_loss, test_mix_sdr, test_sdr, test_sir, test_sar))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # arguments
    parser = ArgParser()
    args = parser.parse_train_arguments()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_incremental_steps = args.num_classes // args.class_num_per_step

    args.id += '-Ours'
    args.id += '-memory_{}_per_class'.format(args.exemplar_num_per_class)
    args.id += '-{}_classes_in_one_step'.format(args.class_num_per_step)
    if args.final_mask_distil:
        args.id += '-lam_mask_distl_{}'.format(args.lam_mask_distl)
    if args.cross_modal_contra:
        args.id += '-lam_ins_contra_{}-lam_cls_contra_{}_new'.format(args.lam_ins_contra, args.lam_cls_contra)

    setup_seed(args.seed)

    print('Model ID: {}'.format(args.id))

    dataset_train = MUSICMix21IncrementalOursDataset(args, split='train')
    dataset_val = MUSICMix21IncrementalOursDataset(args, split='val')
    dataset_test = MUSICMix21IncrementalOursDataset(args, split='test')

    args.ckpt = os.path.join(args.ckpt, args.id)
    if args.mode == 'train':
        makedirs(args.ckpt, remove=True)
    
    for step in range(total_incremental_steps):
        args.vis_val = os.path.join(args.ckpt, 'visualization_val_step_{}/'.format(step))
        args.vis_test = os.path.join(args.ckpt, 'visualization_test_step_{}/'.format(step))

        dataset_train._set_incremental_step_(step)
        dataset_val._set_incremental_step_(step)
        dataset_test._set_incremental_step_(step)
    
        print('Step {} starts....'.format(step))
        main(args, step=step, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test)
