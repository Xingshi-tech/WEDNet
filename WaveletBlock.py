import torch
import torch.nn as nn


class BWT(nn.Module):
    def __init__(self):
        super(BWT, self).__init__()

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4
        x_HD = x1 + x2 - x3 - x4
        x_LV = -x1 + x2 + x3 - x4
        x_LD = x1 - x2 + x3 - x4
        x_HV = -x1 - x2 - x3 - x4
        return torch.cat((x_LL, x_HL, x_LH, x_HH, x_HD, x_LV, x_LD, x_HV), 1)


class IBWT(nn.Module):
    def __init__(self):
        super(IBWT, self).__init__()

    def forward(self, x):
        r = 2
        in_batch, in_channel, in_height, in_width = x.size()
        out_batch, out_channel, out_height, out_width = in_batch, int(
            in_channel / (r ** 3)), r * in_height, r * in_width
        x1 = x[:, 0:out_channel, :, :] / 2
        x2 = x[:, out_channel:out_channel * 2, :, :] / 2
        x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
        x5 = x[:, out_channel * 4:out_channel * 5, :, :] / 2
        x6 = x[:, out_channel * 5:out_channel * 6, :, :] / 2
        x7 = x[:, out_channel * 6:out_channel * 7, :, :] / 2
        x8 = x[:, out_channel * 7:out_channel * 8, :, :] / 2
        h = torch.zeros([out_batch, out_channel, out_height, out_width]).cuda()

        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4 + x5 - x6 + x7 - x8
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4 - x5 + x6 - x7 + x8
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4 - x5 - x6 + x7 + x8
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8

        return h


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


class SALayer(nn.Module):
    def __init__(self, kernel_size=5, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, 1, keepdim=True)
        channel_pool = torch.cat([max_pool, avg_pool], dim=1)
        y = self.conv_du(channel_pool)

        return x * y


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class WaveletBlock(nn.Module):
    def __init__(self, n_feat, o_feat, kernel_size, reduction, bias, act):
        super(WaveletBlock, self).__init__()
        self.bwt = BWT()
        self.ibwt = IBWT()

        modules_body = [
            conv(n_feat * 8, n_feat, kernel_size, bias=bias),
            act,
            conv(n_feat, n_feat * 8, kernel_size, bias=bias)
        ]
        self.body = nn.Sequential(*modules_body)

        self.SAL = SALayer()
        self.CAL = CALayer(n_feat * 8, reduction, bias=bias)

        self.conv1x1 = nn.Conv2d(n_feat * 8, n_feat * 8, kernel_size=1, bias=bias)
        self.conv3x3 = nn.Conv2d(n_feat, o_feat, kernel_size=3, padding=1, bias=bias)
        self.activate = act
        self.conv1x1_final = nn.Conv2d(n_feat, o_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        residual = x

        x_bwt = self.bwt(x)

        res = self.body(x_bwt)

        res = self.SAL(res)
        res = self.CAL(res)

        res = self.conv1x1(res) + x_bwt
        wavelet_path = self.ibwt(res)

        out = self.activate(self.conv3x3(wavelet_path))
        out += self.conv1x1_final(residual)

        return out


if __name__ == '__main__':
    x = torch.randn(32, 64, 100, 100).cuda()
    wb = WaveletBlock(n_feat=64, o_feat=64, kernel_size=3, reduction=8, bias=False, act=nn.PReLU()).cuda()
    res = wb(x)
    print(res.shape)
