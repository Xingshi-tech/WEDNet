import torch
from torch import nn

from WaveletBlock import WaveletBlock


class Hist_adjust(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Hist_adjust, self).__init__()
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.point_conv(input)
        return out


class reweight(nn.Module):
    def __init__(self):
        super().__init__()
        self.reweight = nn.Sequential(
            nn.PReLU(num_parameters=1),
            nn.Dropout(p=0.0, inplace=True),
            nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, x):
        return self.reweight(x)


class FeatureFusionConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.f3 = nn.Sequential(
            nn.Conv2d(64 + 16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False),
            nn.LeakyReLU(),
            reweight()
        )

        self.f5 = nn.Sequential(
            nn.Conv2d(64 + 16, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=8, bias=False),
            nn.LeakyReLU(),
            reweight()
        )

        self.f7 = nn.Sequential(
            nn.Conv2d(64 + 16, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=16, bias=False),
            nn.LeakyReLU(),
            reweight()
        )

        self.f9 = nn.Sequential(
            nn.Conv2d(64 + 16, 64, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4), groups=16, bias=False),
            nn.LeakyReLU(),
            reweight()
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(64 * 4, 64 * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(64 * 2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU()
        )

    def forward(self, x, retouch_image):
        input = torch.cat([x, retouch_image], dim=1)

        f3_res = self.f3(input)
        f5_res = self.f5(input)
        f7_res = self.f7(input)
        f9_res = self.f9(input)
        res = self.fusion(torch.cat([f3_res, f5_res, f7_res, f9_res], dim=1))

        return res


class enhance_net_nopool(nn.Module):

    def __init__(self, nbins):
        super(enhance_net_nopool, self).__init__()

        self.nbins = nbins
        self.relu = nn.LeakyReLU()
        number_f = 16
        self.g_conv1 = Hist_adjust(self.nbins + 1, number_f)
        self.g_conv2 = Hist_adjust(number_f, number_f)
        self.g_conv3 = Hist_adjust(number_f + self.nbins + 1, number_f)
        self.g_conv4 = Hist_adjust(number_f, number_f * 2)
        self.g_conv5 = Hist_adjust(number_f * 2, number_f)
        self.g_conv6 = Hist_adjust(number_f, 8)

        self.featureNet = FeatureFusionConvNet()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU(),
        )

        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU()
        )

        self.conv_1_2 = nn.Sequential(
            nn.Conv2d(32 + 3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU()
        )

        list = []
        for i in range(9):
            list.append(nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))
            list.append(nn.LeakyReLU())

        self.fcs = nn.Sequential(*list)

        self.proj = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1)),
            nn.Dropout(p=0.0, inplace=False),
        )

        self.mlp = nn.Sequential(
            nn.PReLU(num_parameters=1),
            nn.Dropout(p=0.0, inplace=True),
            nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU()
        )

        self.conv_v_1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1, bias=False),
            nn.LeakyReLU()
        )
        self.conv_v_2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1, bias=False),
            nn.LeakyReLU()
        )
        
        self.wb = WaveletBlock(n_feat=64, o_feat=64, kernel_size=3, reduction=16, bias=False, act=nn.PReLU())

    def retouch(self, x, x_r):
        x = x + x_r[:, 0:1, :, :] * (-torch.pow(x, 2) + x)
        x = x + x_r[:, 1:2, :, :] * (-torch.pow(x, 2) + x)
        x = x + x_r[:, 2:3, :, :] * (-torch.pow(x, 2) + x)
        x = x + x_r[:, 3:4, :, :] * (-torch.pow(x, 2) + x)
        x = x + x_r[:, 4:5, :, :] * (-torch.pow(x, 2) + x)
        x = x + x_r[:, 5:6, :, :] * (-torch.pow(x, 2) + x)
        x = x + x_r[:, 6:7, :, :] * (-torch.pow(x, 2) + x)
        x = x + x_r[:, 7:8, :, :] * (-torch.pow(x, 2) + x)

        return x

    def forward(self, x, hist):
        x_V = x.max(1, keepdim=True)[0]

        g1 = self.relu(self.g_conv1(hist))
        g2 = self.relu(self.g_conv2(g1))
        g3 = self.relu(self.g_conv3(torch.cat([g2, hist], 1)))
        g4 = self.relu(self.g_conv4(g3))
        g5 = self.relu(self.g_conv5(g4))

        retouch_image = self.retouch(x_V, g5)

        enhance_v_1 = self.conv_v_1(retouch_image)
        enhance_v_2 = self.conv_v_2(enhance_v_1)

        x1 = self.conv_1(x)
        x1_1 = self.conv_1_1(x1)
        x1_2 = self.conv_1_2(torch.cat([x1_1, x], 1))
        x2 = x1_2 + self.fcs(x1_2)

        fusion = self.featureNet(x2, enhance_v_2)

        wb = self.wb(fusion)

        x9 = x2 + self.proj(wb)
        x10 = self.mlp(x9)
        x11 = self.conv_2(x10)
        return torch.clamp(x11, 0, 1)
