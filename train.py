"""
Train model with argument setting
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

from datetime import datetime
from math import ceil
import os
import os.path as osp
import pickle
import numpy as np
import cv2
import lmdb
import torch
import random
import copy
import utils
from model import STSR, RefineLayer

# Superslomo module
import IT_arch.superslomo as superslomo

# Dataset & loss & optimizer
from dataset import vimeo90k
from radam import RAdam

# Extra module
from tqdm import tqdm
import argparse


##############################
#     Constant Setting       #
##############################
parser = argparse.ArgumentParser()

## Hyper-parameter
parser.add_argument("--gpu_id", type=str, default='0')
parser.add_argument("--batch_size", type=int, default=12)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--dataset", type=str, default='vimeo90k')
parser.add_argument("--data_root", type=str, default='/data')
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--merge_lr", type=float, default=1e-4)

## Model related settings
parser.add_argument("--sr_type", type=str, choices=['ESPCN', 'SAN'])
parser.add_argument("--it_type", type=str, choices=['SSM', 'DAIN'])
parser.add_argument("--merge_in", type=int)
parser.add_argument("--merge_out", type=int)
parser.add_argument("--two_mask", action="store_true")
parser.add_argument("--refine_type", type=str, default='resblock')
parser.add_argument("--refine_in", type=int, default=3)
parser.add_argument("--input_R", type=str, choices=['ST', 'TS', 'Both', 'Half', 'IFISTITS', '3timestamp'], default='Both')
parser.add_argument("--stsr_weight", type=str)

## Training strategy
parser.add_argument("--train_MsMt", action='store_true')
parser.add_argument("--train_F", action='store_true')
parser.add_argument("--train_R", action='store_true')
parser.add_argument("--detach", action='store_true')
parser.add_argument("--seed", type=int, default=0)

## Other option
parser.add_argument("--save_dir", type=str, default='./results/')

args = parser.parse_args()
print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_dir = os.path.join(args.save_dir)
os.makedirs(save_dir, exist_ok=True)
SPATIAL_SIZE = (64, 112)
nn_down = nn.Upsample(scale_factor=0.5, mode='bicubic').to(device)


## randomness
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    # Load dataset & dataloader
    if args.dataset == 'vimeo90k':
        train_dataset = vimeo90k.Vimeo90kDataset(args.data_root, SPATIAL_SIZE, train=True)
        val_dataset = vimeo90k.Vimeo90kDataset(args.data_root, (64, 112), train=False)
    else:
        raise NotImplementedError('Not support other dataset now')

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=24, shuffle=False, num_workers=8, pin_memory=True)
    
    # Load SR/interp model
    stsr = STSR(
        args.sr_type, None,
        args.it_type, None,
        args.merge_in, args.merge_out, None, args.two_mask,
        args.refine_type, args.refine_in, None, args.input_R,
        args.train_MsMt, args.train_F, args.train_R, args.detach
    )

    cri = nn.L1Loss().to(device)

    # set and load optimizer
    optimizer = stsr.get_optimizer(
        args.lr, args.merge_lr, args.train_MsMt, args.train_F, args.train_R
    )

    # Dataparallel and to device
    stsr = nn.DataParallel(stsr).to(device)
    
    _ = val_loop(stsr, val_loader, val_dataset, args.start_epoch)
    best_PSNR = 0
    for epoch in tqdm(range(args.start_epoch, args.epochs), desc='[EPOCH]'):
        record_TOTAL = []
        record_MERGE = []
        record_RESIDUAL = []
        record_SR = []
        record_IT = []
        record_SR_IT = []
        record_IT_SR = []
        record_WARP = []

        stsr.train()
        for _, train_data in enumerate(tqdm(train_loader, desc='[TRAIN]')):
            HR = train_data['HR'].to(device)
            LR = torch.stack([nn_down(HR[:, 0]), nn_down(HR[:, 1]), nn_down(HR[:, 2])], dim=1)
            LR = LR.clamp(0, 1).detach()
            GT = HR[:, 1]
            I_L_2, I_H_1, I_H_3, I_TS_2, I_ST_2, I_F_2, mask_1, mask_2, I_R_basic, I_R_2 = stsr(LR[:, 0], LR[:, 2])

            # calculate and record loss
            loss_total = 0
            if args.train_MsMt:
                loss_it = cri(I_L_2, LR[:, 1])
                loss_sr = 0.5*cri(I_H_1, HR[:, 0]) + 0.5*cri(I_H_3, HR[:, 2])
                loss_it_sr = cri(I_TS_2, HR[:, 1])
                loss_sr_it = cri(I_ST_2, HR[:, 1])
                loss_total += loss_it + loss_sr + loss_it_sr + loss_sr_it
                record_IT.append(loss_it.item())
                record_SR.append(loss_sr.item())
                record_SR_IT.append(loss_sr_it.item())
                record_IT_SR.append(loss_it_sr.item())

            if args.train_F:
                loss_merge = cri(I_F_2, GT)
                loss_total += loss_merge
                record_MERGE.append(loss_merge.item())

            if args.train_R:
                loss_residual = cri(I_R_2, GT)
                loss_total += loss_residual
                record_RESIDUAL.append(loss_residual.item())

            record_TOTAL.append(loss_total.item())

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

        loss_log = {
            'Total Loss': np.mean(record_TOTAL)
        }

        if args.train_MsMt:
            loss_log['IT Loss'] = np.mean(record_IT)
            loss_log['SR Loss'] = np.mean(record_SR)
            loss_log['SR_IT Loss'] = np.mean(record_SR_IT)
            loss_log['IT_SR Loss'] = np.mean(record_IT_SR)
        
        if args.train_F:
            loss_log['MERGE Loss'] = np.mean(record_MERGE)

        if args.train_R:
            loss_log['Residual Loss'] = np.mean(record_RESIDUAL)

        if args.train_warp:
            loss_log['Warp Loss'] = np.mean(record_WARP)
        print(loss_log)

        ### validation
        avg_PSNR = val_loop(stsr, val_loader, val_dataset, epoch+1)
        utils.save_model(stsr, 'STSR', 'current', save_dir)
        if avg_PSNR > best_PSNR:
            best_PSNR = avg_PSNR
            utils.save_model(stsr, 'STSR', 'best', save_dir)

        if (epoch+1) % 5 == 0:
            utils.save_model(stsr, 'STSR', epoch+1, save_dir)
        torch.save(optimizer.state_dict(), os.path.join(save_dir, 'optimizer.pth'))


def val_loop(stsr, val_loader, val_dataset, epoch):
    ### validation
    avg_PSNR_TS = 0
    avg_PSNR_ST = 0
    avg_PSNR_MERGE = 0
    avg_PSNR_RESIDUAL = 0
    avg_PSNR_HR = 0
    avg_PSNR_LR = 0

    stsr.eval()


    with torch.no_grad():
        for vid, val_data in enumerate(tqdm(val_loader)):
            """
            TEST CODE
            """
            if args.train_MsMt:
                HR = val_data['HR'].to(device)
                LR = torch.stack([nn_down(HR[:, 0]), nn_down(HR[:, 1]), nn_down(HR[:, 2])], dim=1)
                LR = LR.clamp(0, 1).detach()
                GT = HR[:, 1]
                I_L_2, I_H_1, I_H_3, I_TS_2, I_ST_2, I_F_2, mask_1, mask_2, I_R_basic, I_R_2 = stsr(LR[:, 0], LR[:, 2])
            else:
                ST = val_data['ST'].to(device)
                TS = val_data['TS'].to(device)
                GT = val_data['GT'].to(device)
                I_L_2, I_H_1, I_H_3, I_TS_2, I_ST_2, I_F_2, mask_1, mask_2, I_R_basic, I_R_2 = stsr(ST, TS)

            B, C, H, W = GT.size()
            

            for b_id in range(B):
                avg_PSNR_MERGE += utils.cal_psnr(I_R_basic[b_id], GT[b_id]).item()
                avg_PSNR_RESIDUAL += utils.cal_psnr(I_R_2[b_id], GT[b_id]).item()
                avg_PSNR_TS += utils.cal_psnr(I_TS_2[b_id], GT[b_id]).item()
                avg_PSNR_ST += utils.cal_psnr(I_ST_2[b_id], GT[b_id]).item()
                if args.train_MsMt:
                    avg_PSNR_HR += utils.cal_psnr(I_H_1[b_id], HR[b_id, 0]).item()+utils.cal_psnr(I_H_3[b_id], HR[b_id, 2]).item()
                    avg_PSNR_LR += utils.cal_psnr(I_L_2[b_id], LR[b_id, 1]).item()


    log = {
        'PSNR_TS': avg_PSNR_TS/len(val_dataset),
        'PSNR_ST': avg_PSNR_ST/len(val_dataset),
        'PSNR_MERGE': avg_PSNR_MERGE/len(val_dataset),
        'PSNR_RESIDUAL': avg_PSNR_RESIDUAL/len(val_dataset)
    }
    if args.train_MsMt:
        log['PSNR_HR'] = avg_PSNR_HR/len(val_dataset)/2.
        log['PSNR_LR'] = avg_PSNR_LR/len(val_dataset)
    print(log)
    
    return avg_PSNR_RESIDUAL/len(val_dataset)



if __name__ == '__main__':
    main()