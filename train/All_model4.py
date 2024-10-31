#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/11 21:28
# @File    : All_model4.py
# @Description :
# 去除前置的事件3D补全网络；
# 事件数据取反得到黑色的
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/9/26 09:10
# @File    : All_model3.py
# @Description : 全部模型，事件数据没有进行取反
#                   不再进行分块输入！！！



import torch.nn.functional as F
import torch.nn.init
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.init as init


class ConvBlock(nn.Module):
    """ in_channels: 输入到该块的通道数。
        out_channels: 输出的通道数。
        n_convs: 块中包含的卷积层数。
        kernel_size: 卷积核的大小。
        stride: 卷积的步长。
        padding: 卷积的填充。
        downsample: 是否在块的末尾应用下采样"""

    def __init__(
            self,
            in_channels,
            out_channels,
            n_convs=3,
            kernel_size=3,
            stride=1,
            padding=1,
            downsample=True,
            dilation=1,
    ):
        super(ConvBlock, self).__init__()
        self.modules = []

        c_in = in_channels
        c_out = out_channels
        # 给这个卷积块的每一层进行添加 这里共三层
        for i in range(n_convs):
            # 卷积层
            self.modules.append(
                nn.Conv2d(
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    dilation=dilation,
                )
            )
            # 批量归一化 规范输出
            self.modules.append(nn.BatchNorm2d(num_features=out_channels))
            # 激活函数
            self.modules.append(nn.LeakyReLU(0.1))
            c_in = c_out

        if downsample:
            # 下采样 额外添加一个卷积层和一个激活函数
            self.modules.append(
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.modules.append(nn.ReLU())
            # self.modules.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 创建容器 形成一个可以执行顺序数据处理的单一模块
        self.model = nn.Sequential(*self.modules)
        self.init_weights()

    def init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.model(x)


class ConvTransBlock(nn.Module):
    """ in_channels: 输入到该块的通道数。
        out_channels: 输出的通道数。
        n_convs: 块中包含的卷积层数。
        kernel_size: 卷积核的大小。
        stride: 卷积的步长。
        padding: 卷积的填充。
        downsample: 是否在块的末尾应用下采样"""

    def __init__(
            self,
            in_channels,
            out_channels,
            n_convs=3,
            kernel_size=3,
            stride=1,
            padding=1,
            upsample=True,
            dilation=1,
    ):
        super(ConvTransBlock, self).__init__()
        self.modules = []

        c_in = in_channels
        c_out = out_channels
        # 给这个卷积块的每一层进行添加 这里共三层
        for i in range(n_convs):
            # 卷积层
            self.modules.append(
                nn.ConvTranspose2d(
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    dilation=dilation,
                )
            )
            # 批量归一化 规范输出
            self.modules.append(nn.BatchNorm2d(num_features=out_channels))
            # 激活函数
            self.modules.append(nn.LeakyReLU(0.1))
            c_in = c_out

        if upsample:
            # 下采样 额外添加一个卷积层和一个激活函数
            self.modules.append(
                nn.ConvTranspose2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.modules.append(nn.ReLU())
            # self.modules.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 创建容器 形成一个可以执行顺序数据处理的单一模块
        self.model = nn.Sequential(*self.modules)
        self.init_weights()

    def init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.model(x)

class model_feature(nn.Module):
    def __init__(self, in_channels=1, out_channels=256):
        super(model_feature, self).__init__()

        ''' conv_bottom
            四个 ConvBlock 实例，用于自下而上的特征提取（即从输入图像的原始像素到高维特征的逐层抽象）'''
        self.conv_down_0 = ConvBlock(
            in_channels=in_channels,
            out_channels=32,
            n_convs=1,
            kernel_size=3,
            padding=0,
            downsample=False,
        )
        self.conv_down_1 = ConvBlock(
            in_channels=32,
            out_channels=64,
            n_convs=1,
            kernel_size=3,
            padding=0,
            downsample=True,
        )
        self.conv_down_2 = ConvBlock(
            in_channels=64,
            out_channels=128,
            n_convs=1,
            kernel_size=3,
            padding=0,
            downsample=True,
        )
        self.conv_down_3 = ConvBlock(
            in_channels=128,
            out_channels=out_channels,
            n_convs=1,
            kernel_size=3,
            padding=0,
            downsample=True,
        )

        self.conv_up_0 = ConvTransBlock(
            in_channels=out_channels,
            out_channels=128,
            n_convs=1,
            kernel_size=5,
            stride=2,
            padding=0,
            upsample=False,
        )
        self.conv_up_1 = ConvTransBlock(
            in_channels=128,
            out_channels=64,
            n_convs=1,
            kernel_size=5,
            stride=2,
            padding=0,
            upsample=False,
        )
        self.conv_up_2 = ConvTransBlock(
            in_channels=64,
            out_channels=32,
            n_convs=1,
            kernel_size=5,
            stride=2,
            padding=0,
            upsample=False,
        )
        self.conv_up_3 = ConvTransBlock(
            in_channels=32,
            out_channels=in_channels,
            n_convs=1,
            kernel_size=3,
            padding=0,
            stride=1,
            upsample=False,
        )

    def forward(self, x):
        x = x[:, :1, ...]
        f0 = self.conv_down_0(x)
        f1 = self.conv_down_1(f0)
        f2 = self.conv_down_2(f1)
        f3 = self.conv_down_3(f2)

        f = self.conv_up_0(f3)
        f = self.conv_up_1(f)
        f = self.conv_up_2(f)
        out = self.conv_up_3(f)

        return out, f0, f1, f2, f3


class model_test(nn.Module):
    def __init__(self, in_channels=1, out_channels=256):
        super(model_test, self).__init__()

        ''' conv_bottom
            四个 ConvBlock 实例，用于自下而上的特征提取（即从输入图像的原始像素到高维特征的逐层抽象）'''
        self.feature0 = model_feature(
            in_channels=1,
            out_channels=256
        )
        self.feature1 = model_feature(
            in_channels=1,
            out_channels=256
        )
        self.con_conv_up_3 = ConvTransBlock(
            in_channels=out_channels,
            out_channels=128,
            n_convs=1,
            kernel_size=5,
            stride=2,
            padding=0,
            upsample=False,
        )
        self.con_conv_up_2 = ConvTransBlock(
            in_channels=128 * 2,
            out_channels=64,
            n_convs=1,
            kernel_size=5,
            stride=2,
            padding=0,
            upsample=False,
        )
        self.con_conv_up_1 = ConvTransBlock(
            in_channels=64 * 2,
            out_channels=32,
            n_convs=1,
            kernel_size=5,
            stride=2,
            padding=0,
            upsample=False,
        )
        self.con_conv_up_0 = ConvTransBlock(
            in_channels=32 * 2,
            out_channels=in_channels,
            n_convs=1,
            kernel_size=3,
            padding=0,
            stride=1,
            upsample=False,
        )

    def forward(self, ir, event):
        out0, i0, i1, i2, i3 = self.feature0(ir)
        out1, e0, e1, e2, e3 = self.feature1(event)

        f3 = self.con_conv_up_3(i3 + e3)
        f2 = self.con_conv_up_2(torch.cat([f3[:,:,:,:84], i2 + e2], dim=1))
        f1 = self.con_conv_up_1(torch.cat([f2[:,:,:128,:], i1 + e1], dim=1))
        output = self.con_conv_up_0(torch.cat([f1[:,:,:258,:344], i0 + e0], dim=1))
        output = torch.sigmoid(output)
        return output
