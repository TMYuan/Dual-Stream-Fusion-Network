import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import model
from skimage.measure import compare_ssim, compare_psnr

# ESPCN module
from SR_arch.espcn import Net as ESPCN

# Superslomo module
import IT_arch.superslomo as superslomo

#SAN module
import SAN
import pickle

# DAIN module
# import IT_arch.DAIN as DAIN_arch


###########################################################
# Model operation
# ##########################################################
def save_model(model, model_name, niter, save_dir):
    save_name = '{}_{}.pth'.format(model_name, niter)
    if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module
    state_dict = model.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, os.path.join(save_dir, save_name))

def load_sr(sr_type, weight_path=None):
    print('Load SR model [{}]'.format(sr_type))
    if weight_path is not None:
        print('Load model from [{}]'.format(weight_path))
    else:
        print('Train from scratch')

    if sr_type == 'ESPCN':
        return load_ESPCN(weight_path=weight_path)
    elif sr_type == 'SAN':
        san_model = load_SAN(weight_path=weight_path)
        return san_model
    else:
        raise NotImplementedError('Only accept [ESPCN] or [SAN]')

def load_ESPCN(weight_path=None):
    # sr_model = ESPCN(upscale_factor=4)
    sr_model = ESPCN(upscale_factor=2)
    if weight_path is not None:
        model_dict = torch.load(weight_path)
        sr_model.load_state_dict(model_dict)
    return sr_model

def load_SAN(weight_path=None):
    with open('./SAN/SAN_opt.pkl', 'rb') as f:
        SAN_opt = pickle.load(f)
    sr_model = SAN.Model(SAN_opt)
    if weight_path is not None:
        model_dict = torch.load(weight_path)
        sr_model.model.load_state_dict(model_dict)
    return sr_model

def load_it(it_type, weight_path=None):
    print('Load IT model [{}]'.format(it_type))
    if weight_path is not None:
        print('Load model from [{}]'.format(weight_path))
    else:
        print('Train from scratch')

    if it_type == 'SSM':
        return load_SSM(weight_path=weight_path)
    # elif it_type == 'DAIN':
    #     return load_DAIN(weight_path=weight_path)
    else:
        raise NotImplementedError('Only accept [SSM] or [DAIN] now')

def load_SSM(weight_path=None):
    it_model = superslomo.SuperSloMo(spatial_size=(256, 448), val_spatial_size=(256, 448))
    if weight_path is not None:
        model_dict = torch.load(weight_path)
        it_model.load_state_dict(model_dict)
    return it_model

# def load_DAIN(weight_path=None):
#     interp_model = DAIN_arch(channel=3, filter_size=4, timestep=0.5, training=True)
#     if weight_path is not None:
#         interp_pretrain = torch.load(weight_path)
#         model_dict = interp_model.state_dict()
#         interp_pretrain = {k: v for k, v in interp_pretrain.items() if k in model_dict}
#         model_dict.update(interp_pretrain)
#         interp_model.load_state_dict(model_dict)
#         interp_pretrain = None
#     return interp_model

def load_merge(in_channel, out_channel, weight_path=None):
    merge_model = superslomo.NewUNet(in_channel, out_channel)
    # merge_model = superslomo.UNet(in_channel, out_channel)
    if weight_path is not None:
        model_dict = torch.load(weight_path)
        merge_model.load_state_dict(model_dict)
    return merge_model

def load_refine(refine_type, in_channel, weight_path=None):
    print('Load refine model [{}]'.format(refine_type))
    if refine_type == 'unet':
        refine_model = superslomo.NewUNet(in_channel, 3)
    elif refine_type == 'resblock':
        refine_model = model.MultipleBasicBlock_4(in_channel, 128)
    elif refine_type == 'resblock_modified':
        refine_model = model.ResidualBlock(in_channel, 128)
    elif refine_type == 'lite_resblock':
        refine_model = model.LiteResblock(in_channel)
    else:
        raise NotImplementedError('Only accept [unet] or [resblock] or [resblock2]')
    if weight_path is not None:
        model_dict = torch.load(weight_path)
        refine_model.load_state_dict(model_dict)
    return refine_model

###########################################################
# Evaluation functions
# ##########################################################
def cal_psnr(img1, img2):
    img1 = img1.clamp(0, 1).clone().detach().permute(1, 2, 0)
    img2 = img2.clamp(0, 1).clone().detach().permute(1, 2, 0)
    mse = torch.mean(torch.tensor((img1 - img2)**2, dtype=torch.float64))
    return 10 * torch.log10((1.0**2)/mse)
    # psnr = compare_psnr(img1, img2, data_range=1.0)
    return psnr

def cal_psnr_img(img1, img2):
    psnr = compare_psnr(img1, img2, data_range=1.0)
    return psnr

def cal_ssim(img1, img2):
    img1 = img1.clamp(0, 1).clone().detach().cpu().permute(1, 2, 0).numpy()
    img2 = img2.clamp(0, 1).clone().detach().cpu().permute(1, 2, 0).numpy()
    # print(img1.shape)
    # print(img2.shape)
    ssim = compare_ssim(img1, img2, data_range=1.0, multichannel=True)
    return ssim

def read_img(filename, grayscale=0):

    ## read image and convert to RGB in [0, 1]

    if grayscale:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise Exception("Image %s does not exist" %filename)

        img = np.expand_dims(img, axis=2)
    else:
        img = cv2.imread(filename)

        if img is None:
            raise Exception("Image %s does not exist" %filename)

        img = img[:, :, ::-1] ## BGR to RGB
    
    img = np.float32(img) / 255.0

    return img

def img2tensor(img):

    img_t = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    img_t = torch.from_numpy(img_t.astype(np.float32))

    return img_t

def tensor2img(img_t):

    # img = img_t.clamp(0, 1).detach().to("cpu").numpy()
    img = img_t.detach().to("cpu").numpy()
    img = np.transpose(img, (1, 2, 0))

    return img

def save_img(img, filename):

    # print("Save %s" %filename)

    if img.ndim == 3:
        img = img[:, :, ::-1] ### RGB to BGR
    
    ## clip to [0, 1]
    img = np.clip(img, 0, 1)

    ## quantize to [0, 255]
    img = np.uint8(img * 255.0)

    cv2.imwrite(filename, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def RGB2YUV(rgb):
     
    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])
     
    yuv = np.dot(rgb,m)
    yuv[:,:,1:]+=128.0
    return yuv

