import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from IQA_pytorch import SSIM

import pytorch_ssim


class L_recon(nn.Module):

    def __init__(self):
        super(L_recon, self).__init__()
        self.ssim_loss = pytorch_ssim.SSIM()

    def forward(self, R_low, high):
        L1 = torch.abs(R_low - high).mean()
        L2 = (1 - self.ssim_loss(R_low, high)).mean()
        return L1, L2


class L_brightness(nn.Module):

    def __init__(self):
        super(L_brightness, self).__init__()
        self.avg = nn.AvgPool2d(kernel_size=8)

    def forward(self, R_low, high):
        bright1 = torch.mean(torch.pow(torch.pow(R_low[:, 2:3] - high[:, 2:3] + 1e-4, 2), 0.5))
        bright2 = torch.mean(torch.pow(torch.pow(self.avg(R_low) - self.avg(high) + 1e-4, 2), 0.5))
        return bright1 + bright2


class L_color_zy(nn.Module):

    def __init__(self):
        super(L_color_zy, self).__init__()

    def forward(self, x, y):
        product_separte_color = (x * y).mean(1, keepdim=True)
        x_abs = (x ** 2).mean(1, keepdim=True) ** 0.5
        y_abs = (y ** 2).mean(1, keepdim=True) ** 0.5
        loss1 = (1 - product_separte_color / (x_abs * y_abs + 0.00001)).mean() \
                + torch.mean(torch.acos(product_separte_color / (x_abs * y_abs + 0.00001)))

        return loss1


class L_grad_cosist(nn.Module):

    def __init__(self):
        super(L_grad_cosist, self).__init__()
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)

    def gradient_of_one_channel(self, x, y):
        D_org_right = F.conv2d(x, self.weight_right, padding="same")
        D_org_down = F.conv2d(x, self.weight_down, padding="same")
        D_enhance_right = F.conv2d(y, self.weight_right, padding="same")
        D_enhance_down = F.conv2d(y, self.weight_down, padding="same")
        return torch.abs(D_org_right), torch.abs(D_enhance_right), torch.abs(D_org_down), torch.abs(D_enhance_down)

    def gradient_Consistency_loss_patch(self, x, y):
        # B*C*H*W
        min_x = torch.abs(x.min(2, keepdim=True)[0].min(3, keepdim=True)[0]).detach()
        min_y = torch.abs(y.min(2, keepdim=True)[0].min(3, keepdim=True)[0]).detach()
        x = x - min_x
        y = y - min_y
        # B*1*1,3
        product_separte_color = (x * y).mean([2, 3], keepdim=True)
        x_abs = (x ** 2).mean([2, 3], keepdim=True) ** 0.5
        y_abs = (y ** 2).mean([2, 3], keepdim=True) ** 0.5
        loss1 = (1 - product_separte_color / (x_abs * y_abs + 0.00001)).mean() + torch.mean(
            torch.acos(product_separte_color / (x_abs * y_abs + 0.00001)))

        product_combine_color = torch.mean(product_separte_color, 1, keepdim=True)
        x_abs2 = torch.mean(x_abs ** 2, 1, keepdim=True) ** 0.5
        y_abs2 = torch.mean(y_abs ** 2, 1, keepdim=True) ** 0.5
        loss2 = torch.mean(1 - product_combine_color / (x_abs2 * y_abs2 + 0.00001)) + torch.mean(
            torch.acos(product_combine_color / (x_abs2 * y_abs2 + 0.00001)))
        return loss1 + loss2

    def forward(self, x, y):
        x_R1, y_R1, x_R2, y_R2 = self.gradient_of_one_channel(x[:, 0:1, :, :], y[:, 0:1, :, :])
        x_G1, y_G1, x_G2, y_G2 = self.gradient_of_one_channel(x[:, 1:2, :, :], y[:, 1:2, :, :])
        x_B1, y_B1, x_B2, y_B2 = self.gradient_of_one_channel(x[:, 2:3, :, :], y[:, 2:3, :, :])
        x = torch.cat([x_R1, x_G1, x_B1, x_R2, x_G2, x_B2], 1)
        y = torch.cat([y_R1, y_G1, y_B1, y_R2, y_G2, y_B2], 1)

        B, C, H, W = x.shape
        loss = self.gradient_Consistency_loss_patch(x, y)
        loss1 = 0
        loss1 += self.gradient_Consistency_loss_patch(x[:, :, 0:H // 2, 0:W // 2], y[:, :, 0:H // 2, 0:W // 2])
        loss1 += self.gradient_Consistency_loss_patch(x[:, :, H // 2:, 0:W // 2], y[:, :, H // 2:, 0:W // 2])
        loss1 += self.gradient_Consistency_loss_patch(x[:, :, 0:H // 2, W // 2:], y[:, :, 0:H // 2, W // 2:])
        loss1 += self.gradient_Consistency_loss_patch(x[:, :, H // 2:, W // 2:], y[:, :, H // 2:, W // 2:])

        return loss


class L_bright_cosist(nn.Module):

    def __init__(self):
        super(L_bright_cosist, self).__init__()

    def gradient_Consistency_loss_patch(self, x, y):
        # B*C*H*W
        min_x = torch.abs(x.min(2, keepdim=True)[0].min(3, keepdim=True)[0]).detach()
        min_y = torch.abs(y.min(2, keepdim=True)[0].min(3, keepdim=True)[0]).detach()
        x = x - min_x
        y = y - min_y
        # B*1*1,3
        product_separte_color = (x * y).mean([2, 3], keepdim=True)
        x_abs = (x ** 2).mean([2, 3], keepdim=True) ** 0.5
        y_abs = (y ** 2).mean([2, 3], keepdim=True) ** 0.5
        loss1 = (1 - product_separte_color / (x_abs * y_abs + 0.00001)).mean() + torch.mean(
            torch.acos(product_separte_color / (x_abs * y_abs + 0.00001)))

        product_combine_color = torch.mean(product_separte_color, 1, keepdim=True)
        x_abs2 = torch.mean(x_abs ** 2, 1, keepdim=True) ** 0.5
        y_abs2 = torch.mean(y_abs ** 2, 1, keepdim=True) ** 0.5
        loss2 = torch.mean(1 - product_combine_color / (x_abs2 * y_abs2 + 0.00001)) + torch.mean(
            torch.acos(product_combine_color / (x_abs2 * y_abs2 + 0.00001)))
        return loss1 + loss2

    def forward(self, x, y):
        B, C, H, W = x.shape
        loss = self.gradient_Consistency_loss_patch(x, y)
        loss1 = 0
        loss1 += self.gradient_Consistency_loss_patch(x[:, :, 0:H // 2, 0:W // 2], y[:, :, 0:H // 2, 0:W // 2])
        loss1 += self.gradient_Consistency_loss_patch(x[:, :, H // 2:, 0:W // 2], y[:, :, H // 2:, 0:W // 2])
        loss1 += self.gradient_Consistency_loss_patch(x[:, :, 0:H // 2, W // 2:], y[:, :, 0:H // 2, W // 2:])
        loss1 += self.gradient_Consistency_loss_patch(x[:, :, H // 2:, W // 2:], y[:, :, H // 2:, W // 2:])

        return loss


class PSNR(nn.Module):
    def __init__(self, max_val=0):
        super().__init__()

        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)

        if mse == 0:
            return 0

        return 10 * torch.log10((1.0 / mse))


ssim = SSIM()
psnr = PSNR()


def validation(model, val_loader):
    ssim = SSIM()
    psnr = PSNR()
    ssim_list = []
    psnr_list = []
    for i, imgs in enumerate(val_loader):
        with torch.no_grad():
            low_img, high_img, hist, img_name = imgs[0].cuda(), imgs[1].cuda(), imgs[2].cuda(), imgs[3]
            enhanced_image = model(low_img, hist)
        ssim_value = ssim(enhanced_image, high_img, as_loss=False).item()
        psnr_value = psnr(enhanced_image, high_img).item()
        ssim_list.append(ssim_value)
        psnr_list.append(psnr_value)

    SSIM_mean = np.mean(ssim_list)
    PSNR_mean = np.mean(psnr_list)
    print('The SSIM Value is:{:6,.3f}'.format(SSIM_mean))
    print('The PSNR Value is:{:6,.3f}'.format(PSNR_mean))
    return SSIM_mean, PSNR_mean
