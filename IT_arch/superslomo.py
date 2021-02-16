import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class down(nn.Module):
    """
    A class for creating neural network blocks containing layers:
    
    Average Pooling --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU
    
    This is used in the UNet Class to create a UNet like NN architecture.

    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `x` to the neural network
        block.
    """


    def __init__(self, inChannels, outChannels, filterSize):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the first convolutional layer.
            outChannels : int
                number of output channels for the first convolutional layer.
                This is also used as input and output channels for the
                second convolutional layer.
            filterSize : int
                filter size for the convolution filter. input N would create
                a N x N filter.
        """


        super(down, self).__init__()
        # Initialize convolutional layers.
        self.conv1 = nn.Conv2d(inChannels,  outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))
        self.conv2 = nn.Conv2d(outChannels, outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))
           
    def forward(self, x):
        """
        Returns output tensor after passing input `x` to the neural network
        block.

        Parameters
        ----------
            x : tensor
                input to the NN block.

        Returns
        -------
            tensor
                output of the NN block.
        """


        # Average pooling with kernel size 2 (2 x 2).
        x = F.avg_pool2d(x, 2)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv2(x), negative_slope = 0.1)
        return x
    
class up(nn.Module):
    """
    A class for creating neural network blocks containing layers:
    
    Bilinear interpolation --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU
    
    This is used in the UNet Class to create a UNet like NN architecture.

    ...

    Methods
    -------
    forward(x, skpCn)
        Returns output tensor after passing input `x` to the neural network
        block.
    """


    def __init__(self, inChannels, outChannels):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the first convolutional layer.
            outChannels : int
                number of output channels for the first convolutional layer.
                This is also used for setting input and output channels for
                the second convolutional layer.
        """

        
        super(up, self).__init__()
        # Initialize convolutional layers.
        self.conv1 = nn.Conv2d(inChannels,  outChannels, 3, stride=1, padding=1)
        # (2 * outChannels) is used for accommodating skip connection.
        self.conv2 = nn.Conv2d(2 * outChannels, outChannels, 3, stride=1, padding=1)
           
    def forward(self, x, skpCn):
        """
        Returns output tensor after passing input `x` to the neural network
        block.

        Parameters
        ----------
            x : tensor
                input to the NN block.
            skpCn : tensor
                skip connection input to the NN block.

        Returns
        -------
            tensor
                output of the NN block.
        """

        # Bilinear interpolation with scaling 2.
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
        # Convolution + Leaky ReLU on (`x`, `skpCn`)
        x = F.leaky_relu(self.conv2(torch.cat((x, skpCn), 1)), negative_slope = 0.1)
        return x



class UNet(nn.Module):
    """
    A class for creating UNet like architecture as specified by the
    Super SloMo paper.
    
    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `x` to the neural network
        block.
    """


    def __init__(self, inChannels, outChannels):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the UNet.
            outChannels : int
                number of output channels for the UNet.
        """

        
        super(UNet, self).__init__()
        # Initialize neural network blocks.
        self.conv1 = nn.Conv2d(inChannels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1 = down(32, 64, 5)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 256, 3)
        self.down4 = down(256, 512, 3)
        self.down5 = down(512, 512, 3)
        self.up1   = up(512, 512)
        self.up2   = up(512, 256)
        self.up3   = up(256, 128)
        self.up4   = up(128, 64)
        self.up5   = up(64, 32)
        self.conv3 = nn.Conv2d(32, outChannels, 3, stride=1, padding=1)
        
    def forward(self, x):
        """
        Returns output tensor after passing input `x` to the neural network.

        Parameters
        ----------
            x : tensor
                input to the UNet.

        Returns
        -------
            tensor
                output of the UNet.
        """


        x  = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope = 0.1)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        x  = self.down5(s5)
        x  = self.up1(x, s5)
        x  = self.up2(x, s4)
        x  = self.up3(x, s3)
        x  = self.up4(x, s2)
        x  = self.up5(x, s1)
        x  = F.leaky_relu(self.conv3(x), negative_slope = 0.1)
        return x

class NewUNet(nn.Module):
    """
    A class for creating UNet like architecture as specified by the
    Super SloMo paper.
    
    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `x` to the neural network
        block.
    """


    def __init__(self, inChannels, outChannels):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the UNet.
            outChannels : int
                number of output channels for the UNet.
        """

        
        super(NewUNet, self).__init__()
        # Initialize neural network blocks.
        self.conv1 = nn.Conv2d(inChannels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1 = down(32, 64, 5)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 256, 3)
        self.down4 = down(256, 512, 3)
        self.down5 = down(512, 512, 3)
        self.up1   = up(512, 512)
        self.up2   = up(512, 256)
        self.up3   = up(256, 128)
        self.up4   = up(128, 64)
        self.up5   = up(64, 32)
        self.conv3 = nn.Conv2d(32, outChannels, 3, stride=1, padding=1)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.fill_(0.5)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        
    def forward(self, x):
        """
        Returns output tensor after passing input `x` to the neural network.

        Parameters
        ----------
            x : tensor
                input to the UNet.

        Returns
        -------
            tensor
                output of the UNet.
        """


        x  = self.conv1(x)
        s1 = self.conv2(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        x  = self.down5(s5)
        x  = self.up1(x, s5)
        x  = self.up2(x, s4)
        x  = self.up3(x, s3)
        x  = self.up4(x, s2)
        x  = self.up5(x, s1)
        x  = self.conv3(x)
        return x


class backWarp(nn.Module):
    """
    A class for creating a backwarping object.

    This is used for backwarping to an image:

    Given optical flow from frame I0 to I1 --> F_0_1 and frame I1, 
    it generates I0 <-- backwarp(F_0_1, I1).

    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
    """


    def __init__(self, W, H):
        """
        Parameters
        ----------
            W : int
                width of the image.
            H : int
                height of the image.
            device : device
                computation device (cpu/cuda). 
        """


        super(backWarp, self).__init__()
        # create a grid
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.W = W
        self.H = H
        self.gridX = torch.tensor(gridX, requires_grad=False)
        self.gridY = torch.tensor(gridY, requires_grad=False)
        
    def forward(self, img, flow, device):
        """
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
        I0  = backwarp(I1, F_0_1)

        Parameters
        ----------
            img : tensor
                frame I1.
            flow : tensor
                optical flow from I0 and I1: F_0_1.

        Returns
        -------
            tensor
                frame I0.
        """


        # Extract horizontal and vertical flows.
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.gridX.unsqueeze(0).expand_as(u).float().to(device) + u
        y = self.gridY.unsqueeze(0).expand_as(v).float().to(device) + v
        # range -1 to 1
        x = 2*(x/self.W - 0.5)
        y = 2*(y/self.H - 0.5)
        # stacking X and Y
        grid = torch.stack((x,y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid)
        return imgOut


# Creating an array of `t` values for the 7 intermediate frames between
# reference frames I0 and I1. 
# t = np.linspace(0.125, 0.875, 7)
# t = torch.from_numpy(t)

def getFlowCoeff (batch, device):
    """
    Gets flow coefficients used for calculating intermediate optical
    flows from optical flows between I0 and I1: F_0_1 and F_1_0.

    F_t_0 = C00 x F_0_1 + C01 x F_1_0
    F_t_1 = C10 x F_0_1 + C11 x F_1_0

    where,
    C00 = -(1 - t) x t
    C01 = t x t
    C10 = (1 - t) x (1 - t)
    C11 = -t x (1 - t)

    Parameters
    ----------
        indices : tensor
            indices corresponding to the intermediate frame positions
            of all samples in the batch.
        device : device
                computation device (cpu/cuda). 

    Returns
    -------
        tensor
            coefficients C00, C01, C10, C11.
    """


    # Convert indices tensor to numpy array
    # ind = indices.detach().cpu().numpy()
    # t = np.linspace(0.125, 0.875, 7)
    # t = torch.from_numpy(t).float().to(device)

    # ind = indices.detach()
    # C11 = C00 = - (1 - (t[ind])) * (t[ind])
    # C01 = (t[ind]) * (t[ind])
    # C10 = (1 - (t[ind])) * (1 - (t[ind]))
    C11 = C00 = - (1 - 0.5) * (0.5)
    C01 = (0.5) * (0.5)
    C10 = (1 - 0.5) * (1 - 0.5)
    C00 = torch.zeros([batch, 1, 1, 1]).float().fill_(C00).to(device)
    C01 = torch.zeros([batch, 1, 1, 1]).float().fill_(C01).to(device)
    C10 = torch.zeros([batch, 1, 1, 1]).float().fill_(C10).to(device)
    C11 = torch.zeros([batch, 1, 1, 1]).float().fill_(C11).to(device)

    # return torch.Tensor(C00)[None, None, None, :].permute(3, 0, 1, 2).to(device), torch.Tensor(C01)[None, None, None, :].permute(3, 0, 1, 2).to(device), torch.Tensor(C10)[None, None, None, :].permute(3, 0, 1, 2).to(device), torch.Tensor(C11)[None, None, None, :].permute(3, 0, 1, 2).to(device)
    # return C00[None, None, None, :].permute(3, 0, 1, 2), C01[None, None, None, :].permute(3, 0, 1, 2), C10[None, None, None, :].permute(3, 0, 1, 2), C11[None, None, None, :].permute(3, 0, 1, 2)
    return C00, C01, C10, C11

def getWarpCoeff (batch, device):
    """
    Gets coefficients used for calculating final intermediate 
    frame `It_gen` from backwarped images using flows F_t_0 and F_t_1.

    It_gen = (C0 x V_t_0 x g_I_0_F_t_0 + C1 x V_t_1 x g_I_1_F_t_1) / (C0 x V_t_0 + C1 x V_t_1)

    where,
    C0 = 1 - t
    C1 = t

    V_t_0, V_t_1 --> visibility maps
    g_I_0_F_t_0, g_I_1_F_t_1 --> backwarped intermediate frames

    Parameters
    ----------
        indices : tensor
            indices corresponding to the intermediate frame positions
            of all samples in the batch.
        device : device
                computation device (cpu/cuda). 

    Returns
    -------
        tensor
            coefficients C0 and C1.
    """


    # Convert indices tensor to numpy array
    # ind = indices.detach().cpu().numpy()
    # t = np.linspace(0.125, 0.875, 7)
    # t = torch.from_numpy(t).float().to(device)

    # ind = indices.detach()
    # C0 = 1 - t[ind]
    # C1 = t[ind]
    C0 = 1 - 0.5
    C1 = 0.5
    C0 = torch.zeros([batch, 1, 1, 1]).float().fill_(C0).to(device)
    C1 = torch.zeros([batch, 1, 1, 1]).float().fill_(C1).to(device)
    # return torch.Tensor(C0)[None, None, None, :].permute(3, 0, 1, 2).to(device), torch.Tensor(C1)[None, None, None, :].permute(3, 0, 1, 2).to(device)
    # return C0[None, None, None, :].permute(3, 0, 1, 2), C1[None, None, None, :].permute(3, 0, 1, 2)
    return C0, C1


class SuperSloMo(nn.Module):
    """
    Overall superslomo module for training forward and return corresponding output
    [TODO]
    1. Parallel setting: related to device
    """
    def __init__(self, spatial_size, val_spatial_size):
        super(SuperSloMo, self).__init__()
        H, W = spatial_size
        H_val, W_val = val_spatial_size
        # self.set_device = device
        self.flow_comp = UNet(6, 4)
        self.arb_interp = UNet(20, 5)
        self.train_backwarp = backWarp(W, H)
        self.val_backwarp = backWarp(W_val, H_val)
        self.vgg = self._create_vgg()

    def _create_vgg(self):
        vgg16 = torchvision.models.vgg16(pretrained=True)
        vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
        for param in vgg16_conv_4_3.parameters():
            param.requires_grad = False

        return vgg16_conv_4_3

    def _calc_loss(self, out, gt):
        self.vgg.eval()
        img_0, img_1, ft, It_ft_0_f, It_ft_1_f, f1_0, f0_1 = out
        device = img_0.device
        rec_loss = F.l1_loss(ft, gt)
        perceptual_loss = F.mse_loss(self.vgg(ft), self.vgg(gt))
        warp_loss = F.l1_loss(It_ft_0_f, gt) + F.l1_loss(It_ft_1_f, gt) + F.l1_loss(self.train_backwarp(img_0, f1_0, device), img_1) + F.l1_loss(self.train_backwarp(img_1, f0_1, device), img_0)
        loss_smooth_1_0 = torch.mean(torch.abs(f1_0[:, :, :, :-1] - f1_0[:, :, :, 1:])) + torch.mean(torch.abs(f1_0[:, :, :-1, :] - f1_0[:, :, 1:, :]))
        loss_smooth_0_1 = torch.mean(torch.abs(f0_1[:, :, :, :-1] - f0_1[:, :, :, 1:])) + torch.mean(torch.abs(f0_1[:, :, :-1, :] - f0_1[:, :, 1:, :]))
        loss_smooth = loss_smooth_1_0 + loss_smooth_0_1

        return [0.8*rec_loss, 0.4*warp_loss, 2e-5*perceptual_loss, 0.004*loss_smooth]
    
    def forward(self, x, gt=None):
        """
        size of x : (B, 2, C, H, W)
        """
        B, N, C, H, W = x.size()
        device = x.device

        # 1. compute flow between two frames
        # 2. get flow coef
        # 3. Calculate intermediate flow
        x = x.view(B, N*C, H, W)
        flow_out = self.flow_comp(x)
        f0_1 = flow_out[:, :2, :, :]
        f1_0 = flow_out[:, 2:, :, :]
        f_coef = getFlowCoeff(B, device)
        ft_0 = f_coef[0] * f0_1 + f_coef[1] * f1_0
        ft_1 = f_coef[2] * f0_1 + f_coef[3] * f1_0

        # Get intermediate frame
        I0, I1 = x[:, :C, :, :], x[:, C:, :, :]
        if self.training:
            It_ft_0 = self.train_backwarp(I0, ft_0, device)
            It_ft_1 = self.train_backwarp(I1, ft_1, device)
        else:
            It_ft_0 = self.val_backwarp(I0, ft_0, device)
            It_ft_1 = self.val_backwarp(I1, ft_1, device)

        # 1. Calculate optical flow residual,
        #    visibility map through all input
        # 2. Extract optical flow residual and v_map
        intrp_out = self.arb_interp(
            torch.cat((I0, I1, f0_1, f1_0, ft_0, ft_1, It_ft_0, It_ft_1), dim=1)
        )
        ft_0_f = intrp_out[:, :2, :, :] + ft_0
        ft_1_f = intrp_out[:, 2:4, :, :] + ft_1
        v_t_0 = F.sigmoid(intrp_out[:, 4:5, :, :])
        v_t_1 = 1 - v_t_0

        # 1. Get final intermediate frame from refined flow
        #    and coef for final result
        # 2. Get final indermediate result
        if self.training:
            It_ft_0_f = self.train_backwarp(I0, ft_0_f, device)
            It_ft_1_f = self.train_backwarp(I1, ft_1_f, device)
        else:
            It_ft_0_f = self.val_backwarp(I0, ft_0_f, device)
            It_ft_1_f = self.val_backwarp(I1, ft_1_f, device)
        w_coef = getWarpCoeff(B, device)
        ft = (w_coef[0]*v_t_0*It_ft_0_f + w_coef[1]*v_t_1*It_ft_1_f) / (w_coef[0]*v_t_0 + w_coef[1]*v_t_1)
        
        # If in training mode, calculate loss
        # if self.training:
        #     out = [I0, I1, ft, It_ft_0_f, It_ft_1_f, f1_0, f0_1]
        #     loss = self._calc_loss(out, gt)
        #     return out, loss
        # else:
        #     return ft
        return ft

if __name__ == "__main__":
    pass
