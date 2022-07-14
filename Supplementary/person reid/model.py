import torch
import torch.nn as nn
import torch.nn.init as init
import functools
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
from IPython import embed
import math


# 功能相当于 (feature * sigmoid)  激活函数
class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


# 功能相当于sigmoid激活函数
class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        init.constant(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, bias=False)


# conv1x1 followed by bn and relu
class Conv1x1BNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, ReLu):
        super(Conv1x1BNReLU, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_planes),
            ReLu())

    def forward(self, x):
        return self.op(x)


# conv1x1 with bn no relu for linear transformation
class Conv1x1BN(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Conv1x1BN, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_planes))

    def forward(self, x):
        return self.op(x)


class ConvBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, keepsame=True):
        super(ConvBNReLU, self).__init__()
        # resize the padding to keep the same shape for special kernel_size
        if keepsame:
            padding = kernel_size // 2
        self.op = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.op(x)


class Spatial(nn.Module):
    def __init__(self):
        super(Spatial, self).__init__()
        self.conv1 = ConvBlock(384, 384, 3, s=2, p=1)
        self.conv2 = ConvBlock(384, 4, 1)
        self.sigmoid = nn.Sigmoid()

        self.conv_1 = ConvBlock(4, 1, 1)
        self.conv_2 = ConvBlock(4, 1, 1)
        self.conv_3 = ConvBlock(4, 1, 1)
        self.conv_4 = ConvBlock(4, 1, 1)

        self.part_num = 4

    def forward(self, x):
        b, c, h, w = x.size()
        fea = x
        x = self.conv1(x)
        x = F.upsample(x, (x.size(2) * 2, x.size(3) * 2), mode='bilinear', align_corners=True)
        x = self.conv2(x)
        w = self.sigmoid(x)

        w_1 = self.conv_1(w)
        w_2 = self.conv_2(w)
        w_3 = self.conv_3(w)
        w_4 = self.conv_4(w)

        w_back = torch.cat((w_1, w_2, w_3, w_4), dim=1)
        w_back = w_back.view(w_back.size(0), w_back.size(1), -1)
        w_back_t = w_back.permute(0, 2, 1)

        loss = torch.bmm(w_back, w_back_t)
        loss = torch.triu(loss, diagonal=1).sum() / (b * self.part_num * (self.part_num - 1) / 2)

        return w_1, w_2, w_3, w_4, loss


class ChannelAttn(nn.Module):
    def __init__(self, in_channels, reduction_rate=16):
        super(ChannelAttn, self).__init__()
        assert in_channels % reduction_rate == 0
        self.conv1 = ConvBlock(in_channels, in_channels // reduction_rate, 1)
        self.conv2 = ConvBlock(in_channels // reduction_rate, in_channels, 1)

    def forward(self, x):
        fea = x
        x = F.avg_pool2d(x, x.size()[2:])
        x = self.conv1(x)
        x = self.conv2(x)
        w = F.sigmoid(x)
        return w * fea


class AdaptiveFuse(nn.Module):
    def __init__(self, in_planes, reduction=4, layer_norm=False):
        super(AdaptiveFuse, self).__init__()
        self.layer_norm = layer_norm
        mid_planes = in_planes // reduction
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, padding=0, bias=True)
        if self.layer_norm:
            self.norm = nn.LayerNorm(mid_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid_planes, in_planes, kernel_size=1, padding=0, bias=True)
        # self.activation = nn.Sigmoid()
        self.activation = hsigmoid()

    def forward(self, x):
        res = self.avgpool(x) + self.maxpool(x)
        res = self.fc1(res)
        if self.layer_norm:
            res = self.norm(res)
        res = self.relu(res)
        res = self.fc2(res)
        w = self.activation(res)
        return x * w


class Prestage(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(Prestage, self).__init__()
        self.conv1 = ConvBNReLU(in_planes, out_planes, kernel_size, stride, padding, keepsame=False)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x) + self.avgpool(x)
        return x


class DWBlock(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(DWBlock, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(), )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class CBlock(nn.Module):
    def __init__(self, in_planes, out_planes, k1, k2, ReLu, reduction=4, adaptionfuse=True):
        super(CBlock, self).__init__()
        self.adaptionfuse = adaptionfuse
        mid_planes = out_planes // reduction
        self.squeeze1 = Conv1x1BNReLU(in_planes, mid_planes, ReLu)
        self.squeeze2 = Conv1x1BNReLU(mid_planes, mid_planes, ReLu)
        self.conv1 = self.make_block_layer(mid_planes, k1)
        self.conv2 = self.make_block_layer(mid_planes, k2)
        if self.adaptionfuse:
            self.adaption = AdaptiveFuse(mid_planes)
        # note the conv1x1 is linear
        self.restore = Conv1x1BN(mid_planes, out_planes)
        # use for identity
        self.expand = None
        # note the conv1x1 is linear
        if in_planes != out_planes:
            self.expand = Conv1x1BN(in_planes, out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        # reduction for bottleneck
        x = self.squeeze1(x)
        x = self.squeeze2(x)

        res1 = self.conv1(x)
        res2 = self.conv2(x)
        if self.adaptionfuse:
            add = self.adaption(res1) + self.adaption(res2)
        else:
            add = res1 + res2
        res = self.restore(add)
        if self.expand is not None:
            identity = self.expand(identity)
        out = res + identity
        return self.relu(out)

    def make_block_layer(self, in_planes, k):
        blocks = []
        # compute how many dw_conv3x3 to be construct
        num = k // 2
        # blocks.append(conv1x1(in_planes, in_planes))
        for i in range(num):
            blocks.append(DWBlock(in_planes, in_planes))
        return nn.Sequential(*blocks)


class DownSample(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(DownSample, self).__init__()
        self.avg_pool = nn.AvgPool2d(2, stride=stride, padding=0)
        self.max_pool = nn.MaxPool2d(2, stride=stride, padding=0)
        self.conv1x1 = Conv1x1BNReLU(in_planes, out_planes, nn.ReLU)

    # here we do conv1x1 first before avg_pool as osnet
    def forward(self, x):
        x = self.conv1x1(x)
        x = self.avg_pool(x) + self.max_pool(x)
        return x


op_names = ["combined3x5", "combined3x7", "combined5x7", "combined3x9", "combined5x9", "combined7x9"]
kernels = [[3, 5], [3, 7], [5, 7], [3, 9], [5, 9], [7, 9]]


class Base(nn.Module):
    def __init__(self, in_planes, out_planes, ReLu):
        super(Base, self).__init__()
        self.cblock1 = CBlock(in_planes, out_planes, 3, 5, ReLu)
        self.cblock2 = CBlock(in_planes, out_planes, 5, 7, ReLu)

    def forward(self, x):
        x1 = self.cblock1(x)
        x2 = self.cblock2(x)
        x = x1 + x2
        return x


class Light(nn.Module):
    def __init__(self, modelsize, class_num, test):
        super(Light, self).__init__()

        if modelsize == 'L':
            self.size = 256
        if modelsize == 'M':
            self.size = 384
        if modelsize == 'S':
            self.size = 512
        if modelsize == 'XS':
            self.size = 64
        # if modelsize == 'XXS':
        #     self.size = 32

        self.test = test
        self.kaiming_init_()

        self.stem = Prestage(3, self.size, kernel_size=7, stride=2, padding=3)
        self.downsample1 = DownSample(64, 256)
        self.downsample2 = DownSample(256, 384)

        self.Stage1 = nn.Sequential(Base(64, 128, nn.ReLU), Base(128, 64, nn.ReLU))
        self.Stage2 = nn.Sequential(Base(256, 320, nn.ReLU), Base(320, 256, nn.ReLU))
        self.Stage3 = nn.Sequential(Base(384, 448, nn.ReLU6), Base(448, 384, nn.ReLU6))

        self.averagepool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool1 = nn.AdaptiveMaxPool2d((1, 1))

        self.averagepool2 = nn.AdaptiveAvgPool2d((2, 2))
        self.maxpool2 = nn.AdaptiveMaxPool2d((2, 2))

        self.classifier1 = nn.Linear(384, class_num)
        # self.classifier2 = nn.Linear(1536, class_num)

        self.bottleneck1 = nn.BatchNorm1d(384)
        self.bottleneck1.bias.requires_grad_(False)  # no shift

        self.bottleneck1.apply(weights_init_kaiming)
        self.classifier1.apply(weights_init_classifier)

        self.position = Spatial()

        self.conv = nn.Sequential(nn.Conv2d(1536, 512, kernel_size=1),
                                  nn.BatchNorm2d(512),
                                  nn.ReLU())

    def forward(self, x):
        x0 = self.stem(x)

        x1 = self.Stage1(x0)
        fmap1 = self.downsample1(x1)

        x2 = self.Stage2(fmap1)
        fmap2 = self.downsample2(x2)

        x3 = self.Stage3(fmap2)

        fmap3_a = self.averagepool1(x3)
        fmap3_m = self.maxpool1(x3)
        fmap3_add = fmap3_a + fmap3_m

        fmap3_a2 = self.averagepool2(x3)
        fmap3_m2 = self.maxpool2(x3)
        fmap3_add2 = fmap3_a2 + fmap3_m2

        fmap3_af = fmap3_a.view(fmap3_a.size(0), -1)  # triplet loss
        fmap3_mf = fmap3_m.view(fmap3_m.size(0), -1)  # triplet loss
        fmap3_addf1 = fmap3_add.view(fmap3_add.size(0), -1)  # triplet loss

        fmap3_af2 = fmap3_a2.view(fmap3_a2.size(0), -1)  # triplet loss
        fmap3_mf2 = fmap3_m2.view(fmap3_m2.size(0), -1)  # triplet loss
        fmap3_addf2 = fmap3_add2.view(fmap3_add2.size(0), -1)  # triplet loss

        fmap3_addf = self.bottleneck1(fmap3_addf1)
        pred1 = self.classifier1(fmap3_addf)  # id loss
        # pred2 = self.classifier2(fmap3_addf2)  # id loss

        w_1, w_2, w_3, w_4, loss_pos = self.position(x3)
        x_1 = self.averagepool1(w_1 * x3) + self.maxpool1(w_1 * x3)
        x_2 = self.averagepool1(w_2 * x3) + self.maxpool1(w_2 * x3)
        x_3 = self.averagepool1(w_3 * x3) + self.maxpool1(w_3 * x3)
        x_4 = self.averagepool1(w_4 * x3) + self.maxpool1(w_4 * x3)
        #
        # x_1 = x_1.view(x_1.size(0), -1)
        # x_2 = x_2.view(x_2.size(0), -1)
        # x_3 = x_3.view(x_3.size(0), -1)
        # x_4 = x_4.view(x_4.size(0), -1)

        x_toal = torch.cat((x_1, x_2, x_3, x_4), dim=1)

        x_toal = self.conv(x_toal)
        x_toal = x_toal.view(x_toal.size(0), -1)

        if self.test == True:
            return fmap3_addf1
        return pred1, fmap3_af, fmap3_mf, fmap3_af2, fmap3_mf2, fmap3_addf1, fmap3_addf2, loss_pos, x_toal

    def kaiming_init_(self):
        # print("use kaiming init")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    init.constant_(m.weight, 1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


if __name__ == '__main__':
    x = torch.Tensor(32, 3, 256, 128)
    model = Light('XS', 751, test=False)
    y = model(x)

