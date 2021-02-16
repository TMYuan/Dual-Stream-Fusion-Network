import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import torchvision.models as models
from torch.optim import Adam
from Resblock import BasicBlock
from radam import RAdam
import math
import torch.nn.init as weight_init

# Superslomo module
import IT_arch.superslomo as superslomo


class STSR(nn.Module):
    def __init__(
            self,
            sr_type, sr_weight,
            it_type, it_weight,
            merge_in, merge_out, merge_weight, two_mask,
            refine_type, refine_in, refine_weight, input_R,
            train_MsMt, train_F, train_R, detach
        ):
        super(STSR, self).__init__()
        self.sr_type = sr_type
        self.it_type = it_type
        self.sr_model = utils.load_sr(sr_type, sr_weight)
        self.it_model = utils.load_it(it_type, it_weight)
        self.merge_model = utils.load_merge(merge_in, merge_out, merge_weight)
        self.refine_model = utils.load_refine(refine_type, refine_in, refine_weight)

        self.two_mask = two_mask
        self.refine_type = refine_type
        self.refine_in = refine_in
        self.input_R = input_R

        self.low_warp = superslomo.backWarp(224, 128)
        self.high_warp = superslomo.backWarp(448, 256)

        self.train_MsMt = train_MsMt
        self.train_F = train_F
        self.train_R = train_R
        self.detach = detach

    def get_optimizer(self, lr, merge_lr, train_MsMt, train_F, train_R):
        param_list = []
        if train_MsMt:
            param_list.append({'params': self.sr_model.parameters(), 'lr': 1e-5})
            param_list.append({'params': self.it_model.parameters(), 'lr': lr, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 5e-4})
            print('Enable training of MsMt')
        if train_F:
            param_list.append({'params': self.merge_model.parameters(), 'lr': merge_lr})
            print('Enable training of F')
        if train_R:
            param_list.append({'params': self.refine_model.parameters(), 'lr': merge_lr})
            print('Enable training of R')

        optimizer = RAdam(param_list)
        # optimizer = Adam(param_list)
        return optimizer

    def forward(self, x_1, x_3):
        # T -> S path
        # disable temporary
        # with torch.no_grad():
        #     self.sr_model.eval()
        #     self.it_model.eval()
        if self.train_MsMt:
            # now x_1, x_3 is LR
            if self.it_type == 'SSM':
                x_ts = torch.stack([x_1, x_3], dim=1)
                I_L_2 = self._forward_SSM(self.it_model, x_ts)
            elif self.it_type == 'DAIN':
                x_ts = torch.stack([x_1, x_3], dim=1)
                I_L_2 = self._forward_DAIN(self.it_model, x_ts)
            I_TS_2 = self.sr_model(I_L_2)

            # S -> T path
            I_H_1 = self.sr_model(x_1)
            I_H_3 = self.sr_model(x_3)
            I_H_in = torch.stack([I_H_1, I_H_3], dim=1)
            if self.it_type == 'SSM':
                I_ST_2 = self._forward_SSM(self.it_model, I_H_in)
            elif self.it_type == 'DAIN':
                I_ST_2 = self._forward_DAIN(self.it_model, I_H_in)
        else:
            # if not train MsMt; x_1 x_3 are ST TS
            I_L_2 = None
            I_H_1 = None
            I_H_3 = None
            I_ST_2 = x_1
            I_TS_2 = x_3

        if self.detach:
            # detach F input if flag is True
            merge_in = [I_TS_2.detach(), I_ST_2.detach()]
        else:
            merge_in = [I_TS_2, I_ST_2]

        # Merge Layer
        if self.train_F:
            if self.two_mask:
                mask = F.sigmoid(self._forward_merge(self.merge_model, merge_in))
                mask_1 = mask[:, :1]
                mask_2 = mask[:, 1:]
                # I_F_2 = mask_1 * I_TS_2 + mask_2 * I_ST_2
            else:
                mask_1 = F.sigmoid(self._forward_merge(self.merge_model, merge_in))
                mask_2 = (1 - mask_1)
                # I_F_2 = mask_1 * I_TS_2 + mask_2 * I_ST_2
            I_F_2 = mask_1 * merge_in[0] + mask_2 * merge_in[1]
        else:
            mask_1 = None
            mask_2 = None
            I_F_2 = None

        # Refine layer
        if self.input_R == 'ST':
            I_R_basic = I_ST_2.clone()
        elif self.input_R == 'TS':
            I_R_basic = I_TS_2.clone()
        elif self.input_R == 'Both':
            I_R_basic = I_F_2.clone()
        elif self.input_R == 'Half':
            I_R_basic = 0.5*I_TS_2 + 0.5*I_ST_2
        elif self.input_R == '3timestamp':
            I_R_basic = I_F_2.clone()
        else:
            raise NotImplementedError('input_R must be ST, TS, both or Half')

        if self.detach:
            # detach R input if flag is True
            I_R_basic = I_R_basic.detach()

        if self.train_R:
            if self.input_R == '3timestamp':
                # use t+1, t-1 as reference
                I_R_2 = I_R_basic + self.refine_model(torch.cat([I_R_basic, I_H_1.detach(), I_H_3.detach()], dim=1))
            else:
                I_R_2 = I_R_basic + self.refine_model(I_R_basic)
            # else:
            #     raise NotImplementedError('Only accept [unet] or [resblock] or [resblock_modified]')
        else:
            I_R_2 = None

        return I_L_2, I_H_1, I_H_3, I_TS_2, I_ST_2, I_F_2, mask_1, mask_2, I_R_basic, I_R_2

    def _forward_DAIN(self, model, x_in):
        B, N, C, intHeight, intWidth = x_in.size()

        # Only pad necessary
        if intWidth != ((intWidth >> 6) << 6):
            intWidth_pad = (((intWidth >> 6) + 1) << 6)
            intPaddingLeft = int(( intWidth_pad - intWidth)/2)
            intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
        else:
            intWidth_pad = intWidth
            intPaddingLeft = 0
            intPaddingRight= 0

        if intHeight != ((intHeight >> 6) << 6):
            intHeight_pad = (((intHeight >> 6) + 1) << 6)
            intPaddingTop = int((intHeight_pad - intHeight) / 2)
            intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
        else:
            intHeight_pad = intHeight
            intPaddingTop = 0
            intPaddingBottom = 0
        pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])
        
        x_in = pader(x_in.contiguous().view(-1, C, intHeight, intWidth))
        x_in = x_in.view(B, N, C, x_in.size(2), x_in.size(3))
        out = model(x_in)
        return out[:, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth]

    def _forward_SSM(self, model, x_in):
        B, N, C, intHeight, intWidth = x_in.size()

        # Only pad necessary
        if intWidth != ((intWidth >> 5) << 5):
            intWidth_pad = (((intWidth >> 5) + 1) << 5)
            intPaddingLeft = int(( intWidth_pad - intWidth)/2)
            intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
        else:
            intWidth_pad = intWidth
            intPaddingLeft = 0
            intPaddingRight= 0

        if intHeight != ((intHeight >> 5) << 5):
            intHeight_pad = (((intHeight >> 5) + 1) << 5)
            intPaddingTop = int((intHeight_pad - intHeight) / 2)
            intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
        else:
            intHeight_pad = intHeight
            intPaddingTop = 0
            intPaddingBottom = 0
        pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])
        
        x_in = pader(x_in.contiguous().view(-1, C, intHeight, intWidth))
        x_in = x_in.view(B, N, C, x_in.size(2), x_in.size(3))
        self.it_model.train_backwarp = superslomo.backWarp(x_in.size(4), x_in.size(3))
        self.it_model.val_backwarp = superslomo.backWarp(x_in.size(4), x_in.size(3))
        
        out = model(x_in)
        return out[:, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth]

    def _forward_merge(self, model, x_in):
        x_1 = x_in[0]
        x_2 = x_in[1]

        B, C, intHeight, intWidth = x_1.size()
        # Only pad necessary
        if intWidth != ((intWidth >> 5) << 5):
            intWidth_pad = (((intWidth >> 5) + 1) << 5)
            intPaddingLeft = int(( intWidth_pad - intWidth)/2)
            intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
        else:
            intWidth_pad = intWidth
            intPaddingLeft = 0
            intPaddingRight= 0

        if intHeight != ((intHeight >> 5) << 5):
            intHeight_pad = (((intHeight >> 5) + 1) << 5)
            intPaddingTop = int((intHeight_pad - intHeight) / 2)
            intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
        else:
            intHeight_pad = intHeight
            intPaddingTop = 0
            intPaddingBottom = 0
        pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])
        
        x_1 = pader(x_1.contiguous())
        x_2 = pader(x_2.contiguous())
        
        out = model(torch.cat([x_1, x_2], dim=1))
        return out[:, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth]


def conv3x3(in_planes, out_planes, dilation = 1, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=int(dilation*(3-1)/2), dilation=dilation, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, dilation = 1, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes,dilation, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #         # weight_init.xavier_normal()
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class MultipleBasicBlock(nn.Module):

    def __init__(self,input_feature,
                 block, num_blocks,
                 intermediate_feature = 64, dense = True):
        super(MultipleBasicBlock, self).__init__()
        self.dense = dense
        self.num_block = num_blocks
        self.intermediate_feature = intermediate_feature

        self.block1= nn.Sequential(*[
            nn.Conv2d(input_feature, intermediate_feature,
                      kernel_size=7, stride=1, padding=3, bias=True),
            nn.ReLU(inplace=True)
        ])

        # for i in range(1, num_blocks):
        self.block2 = block(intermediate_feature, intermediate_feature, dilation = 1) if num_blocks>=2 else None
        self.block3 = block(intermediate_feature, intermediate_feature, dilation = 1) if num_blocks>=3 else None
        self.block4 = block(intermediate_feature, intermediate_feature, dilation = 1) if num_blocks>=4 else None
        self.block5 = nn.Sequential(*[nn.Conv2d(intermediate_feature, 3 , (3, 3), 1, (1, 1))])

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x) if self.num_block>=2 else x
        x = self.block3(x) if self.num_block>=3 else x
        x = self.block4(x) if self.num_block== 4 else x
        x = self.block5(x)
        return x

def MultipleBasicBlock_4(input_feature,intermediate_feature = 64):
    model = MultipleBasicBlock(input_feature,
                               BasicBlock,4 ,
                               intermediate_feature)
    return model

class ResidualBlock(nn.Module):
    def __init__(self, in_feature, middle_feature=128):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_feature, middle_feature, 1, 1)
        self.conv2 = conv3x3(middle_feature, middle_feature, 1, 1)
        self.conv3 = conv3x3(middle_feature, middle_feature, 1, 1)
        self.conv4 = conv3x3(middle_feature, middle_feature, 1, 1)
        self.conv5 = conv3x3(middle_feature, in_feature, 1, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return x + out

class LiteResblock(nn.Module):
    def __init__(self, in_feature):
        super(LiteResblock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_feature, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.conv7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.conv8 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        out = self.conv8(self.conv7(self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x))))))))
        # out += x 
        return out
