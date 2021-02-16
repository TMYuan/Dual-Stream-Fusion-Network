"""
Test the given models on vimeo-90k test dataset
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import os.path as osp
import numpy as np
import torch
import utils
from model import STSR

# Dataset & loss & optimizer
from dataset import vimeo90k

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
parser.add_argument("--dataset", type=str, default='vimeo90k')
parser.add_argument("--data_root", type=str, default='/work/minyuan000/vimeo90k/')

## Model related settings
parser.add_argument("--sr_type", type=str, choices=['ESPCN', 'SAN'], default='ESPCN')
parser.add_argument("--sr_weight", type=str)
parser.add_argument("--it_type", type=str, choices=['SSM', 'DAIN'], default='SSM')
parser.add_argument("--it_weight", type=str)
parser.add_argument("--merge_in", type=int, default=6)
parser.add_argument("--merge_out", type=int, default=2)
parser.add_argument("--merge_weight", type=str)
parser.add_argument("--two_mask", action='store_true')
parser.add_argument("--refine_type", type=str, choices=['unet', 'resblock', 'resblock_modified', 'lite_resblock'], default='resblock')
parser.add_argument("--refine_in", type=int, default=3)
parser.add_argument("--input_R", type=str, choices=['ST', 'TS', 'Both', 'Half', 'IFISTITS', '3timestamp'], default='Both')
parser.add_argument("--stsr_weight", type=str)

## Training strategy
parser.add_argument("--forward_MsMt", action='store_true')
parser.add_argument("--forward_F", action='store_true')
parser.add_argument("--forward_R", action='store_true')
parser.add_argument("--detach", action='store_true')

## Other options
parser.add_argument("--name", type=str, default='EVALUATE MODEL')
parser.add_argument("--save_dir", type=str, default='./results/')
args = parser.parse_args()

print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SPATIAL_SIZE = (64, 112)
save_dir = os.path.join(args.save_dir)
os.makedirs(save_dir, exist_ok=True)
nn_down = nn.Upsample(scale_factor=0.5, mode='bicubic').to(device)

def main():
    # Load dataset & dataloader
    if args.dataset == 'vimeo90k':
        train_dataset = vimeo90k.Vimeo90kDataset(args.data_root, SPATIAL_SIZE, train=True)
        val_dataset = vimeo90k.Vimeo90kDataset(args.data_root, (64, 112), train=False)
    else:
        raise NotImplementedError('unknown dataset type')
    
    print('Number of validation data: {}'.format(len(val_dataset)))
    val_loader = DataLoader(dataset=val_dataset, batch_size=20, shuffle=False, num_workers=8, pin_memory=True)

    # Load SR/interp model
    stsr = STSR(
        args.sr_type, None,
        args.it_type, None,
        args.merge_in, args.merge_out, args.merge_weight, args.two_mask,
        args.refine_type, args.refine_in, None, args.input_R,
        args.forward_MsMt, args.forward_F, args.forward_R, args.detach
    )

    if args.stsr_weight is not None:
        stsr.load_state_dict(torch.load(args.stsr_weight))

    if args.sr_weight is not None and args.it_weight is not None:
        stsr.sr_model = utils.load_sr(args.sr_type, args.sr_weight)
        stsr.it_model = utils.load_it(args.it_type, args.it_weight)
        stsr.train_MsMt = True
        args.forward_MsMt = True

    stsr = stsr.to(device).eval()
    
    _ = val_loop(stsr, val_loader, val_dataset, 0)

def val_loop(stsr, val_loader, val_dataset, epoch):
    ### validation
    avg_PSNR_TS = 0
    avg_PSNR_ST = 0
    avg_PSNR_MERGE = 0
    avg_PSNR_RESIDUAL = 0
    avg_PSNR_HR = 0
    avg_PSNR_LR = 0

    avg_SSIM_TS = 0
    avg_SSIM_ST = 0
    avg_SSIM_MERGE = 0
    avg_SSIM_RESIDUAL = 0
    avg_SSIM_HR = 0
    avg_SSIM_LR = 0

    stsr.eval()

    n = 0
    with torch.no_grad():
        # for vid, val_data in enumerate(tqdm(val_loader)):
        for vid, val_data in enumerate(val_loader):
            """
            TEST CODE
            """
            if args.forward_MsMt:
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

                avg_PSNR_TS += utils.cal_psnr(I_TS_2[b_id], HR[b_id, 1]).item()
                avg_PSNR_ST += utils.cal_psnr(I_ST_2[b_id], HR[b_id, 1]).item()
                avg_PSNR_MERGE += utils.cal_psnr(I_F_2[b_id], HR[b_id, 1]).item()
                avg_PSNR_RESIDUAL += utils.cal_psnr(I_R_2[b_id], HR[b_id, 1]).item()
                avg_PSNR_HR += utils.cal_psnr(I_H_1[b_id], HR[b_id, 0]).item()+utils.cal_psnr(I_H_3[b_id], HR[b_id, 2]).item()
                avg_PSNR_LR += utils.cal_psnr(I_L_2[b_id], LR[b_id, 1]).item()

                avg_SSIM_TS += utils.cal_ssim(I_TS_2[b_id], HR[b_id, 1])
                avg_SSIM_ST += utils.cal_ssim(I_ST_2[b_id], HR[b_id, 1])
                avg_SSIM_MERGE += utils.cal_ssim(I_F_2[b_id], HR[b_id, 1])
                avg_SSIM_RESIDUAL += utils.cal_ssim(I_R_2[b_id], HR[b_id, 1])
                avg_SSIM_HR += utils.cal_ssim(I_H_1[b_id], HR[b_id, 0])+utils.cal_ssim(I_H_3[b_id], HR[b_id, 2])
                avg_SSIM_LR += utils.cal_ssim(I_L_2[b_id], LR[b_id, 1])


    f = open(os.path.join(save_dir, 'vimeo_record.txt'), 'w')
    print('PSNR_HR: {}'.format(avg_PSNR_HR/len(val_dataset)/2), file=f)
    print('PSNR_LR: {}'.format(avg_PSNR_LR/len(val_dataset)), file=f)
    print('PSNR_TS: {}'.format(avg_PSNR_TS/len(val_dataset)), file=f)
    print('PSNR_ST: {}'.format(avg_PSNR_ST/len(val_dataset)), file=f)
    print('PSNR_MERGE: {}'.format(avg_PSNR_MERGE/len(val_dataset)), file=f)
    print('PSNR_REFINE: {}'.format(avg_PSNR_RESIDUAL/len(val_dataset)), file=f)

    print('SSIM_HR: {}'.format(avg_SSIM_HR/len(val_dataset)/2), file=f)
    print('SSIM_LR: {}'.format(avg_SSIM_LR/len(val_dataset)), file=f)
    print('SSIM_TS: {}'.format(avg_SSIM_TS/len(val_dataset)), file=f)
    print('SSIM_ST: {}'.format(avg_SSIM_ST/len(val_dataset)), file=f)
    print('SSIM_MERGE: {}'.format(avg_SSIM_MERGE/len(val_dataset)), file=f)
    print('SSIM_REFINE: {}'.format(avg_SSIM_RESIDUAL/len(val_dataset)), file=f)
    f.close()



if __name__ == '__main__':
    main()