import torch
import torch.nn as nn
import torch.nn.functional as F

class Rgb2Ycbcr(nn.Module):
    def __init__(self):
        super(Rgb2Ycbcr, self).__init__()
    
    def forward(self, input):
        device = input.device
        temp = input.clone().float()
        temp *= 255.
        output = torch.empty(input.size())
        output[:, 0, :, :] = (temp[:, 0, :, :] * 65.481 + temp[:, 1, :, :] * 128.553 + temp[:, 2, :, :] * 24.966) / 255. + 16
        output[:, 1, :, :] = (temp[:, 0, :, :] * -37.797 + temp[:, 1, :, :] * -74.203 + temp[:, 2, :, :] * 112.0) / 255. + 128
        output[:, 2, :, :] = (temp[:, 0, :, :] * 112.0 + temp[:, 1, :, :] * -93.786 + temp[:, 2, :, :] * -18.214) / 255. + 128
        output /= 255.
        return output.to(device)

class Ycbcr2Rgb(nn.Module):
    def __init__(self):
        super(Ycbcr2Rgb, self).__init__()
    
    def forward(self, input):
        device = input.device
        temp = input.clone().float()
        temp *= 255.
        output = torch.empty(input.size())
        output[:, 0, :, :] = (temp[:, 0, :, :] * 0.00456621 + temp[:, 1, :, :] * 0 + temp[:, 2, :, :] * 0.00625893) * 255. + -222.921
        output[:, 1, :, :] = (temp[:, 0, :, :] * 0.00456621 + temp[:, 1, :, :] * -0.00153632 + temp[:, 2, :, :] * -0.00318811) * 255. + 135.576
        output[:, 2, :, :] = (temp[:, 0, :, :] * 0.00456621 + temp[:, 1, :, :] * 0.00791071 + temp[:, 2, :, :] * 0) * 255. + -276.836
        output /= 255.
        return output.to(device)
        

class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='bicubic')
        self.rgb2ycbcr = Rgb2Ycbcr()
        self.ycbcr2rgb = Ycbcr2Rgb()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        # self.conv1 = nn.Conv2d(3, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        # self.conv4 = nn.Conv2d(32, 3 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, 1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, img):
        img_ycbcr = self.rgb2ycbcr(img)
        out_y = F.tanh(self.conv1(img_ycbcr[:, :1, :, :]))
        out_y = F.tanh(self.conv2(out_y))
        out_y = F.tanh(self.conv3(out_y))
        out_y = F.sigmoid(self.pixel_shuffle(self.conv4(out_y)))
        out_cbcr = self.upsample(img_ycbcr[:, 1:, :, :])
        out = self.ycbcr2rgb(torch.cat((out_y, out_cbcr), dim = 1))
        return out


# import torch.nn as nn
# import torch.nn.functional as F


# class Net(nn.Module):
#     def __init__(self, upscale_factor):
#         super(Net, self).__init__()

#         self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
#         # self.conv1 = nn.Conv2d(3, 64, (5, 5), (1, 1), (2, 2))
#         self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
#         self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
#         # self.conv4 = nn.Conv2d(32, 3 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
#         self.conv4 = nn.Conv2d(32, 1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

#     def forward(self, x):
#         x = F.tanh(self.conv1(x))
#         x = F.tanh(self.conv2(x))
#         x = F.tanh(self.conv3(x))
#         x = F.sigmoid(self.pixel_shuffle(self.conv4(x)))
#         return x


# if __name__ == "__main__":
#     model = Net(upscale_factor=3)
#     print(model)