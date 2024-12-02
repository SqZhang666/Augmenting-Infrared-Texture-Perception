#!/usr/bin/env python

import getopt
import math
import numpy
import PIL
import PIL.Image
import sys
import torch

try:
    from .correlation import correlation # the custom cost volume layer
except:
    sys.path.insert(0, './correlation'); import correlation # you should consider upgrading python
# end

##########################################################

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

args_strModel = 'default' # 'default', or 'chairs-things'
args_strOne = './images/one.png'
args_strTwo = './images/two.png'
args_strOut = './out.flo'

for strOption, strArg in getopt.getopt(sys.argv[1:], '', [
    'model=',
    'one=',
    'two=',
    'out=',
])[0]:
    if strOption == '--model' and strArg != '': args_strModel = strArg # which model to use
    if strOption == '--one' and strArg != '': args_strOne = strArg # path to the first frame
    if strOption == '--two' and strArg != '': args_strTwo = strArg # path to the second frame
    if strOption == '--out' and strArg != '': args_strOut = strArg # path to where the output should be stored
# end

##########################################################

backwarp_tenGrid = {}
backwarp_tenPartial = {}


def backwarp(tenInput, tenFlow):
    """
    使用光流反向变换一个输入图像的像素位置。

    参数：
    - tenInput (torch.Tensor): 需要反向映射的输入特征图或图像。
    - tenFlow (torch.Tensor): 光流张量，包含了每个像素的运动向量。

    返回：
    - torch.Tensor: 经过光流反向变换后的图像或特征图。
    """

    # 检查并缓存网格
    if str(tenFlow.shape) not in backwarp_tenGrid:
        # 构建水平和垂直的标准化网格
        tenHor = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])

        # 将网格缓存，避免重复创建
        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1).cuda()

    # 为输入图像创建填充值的张量
    if str(tenFlow.shape) not in backwarp_tenPartial:
        backwarp_tenPartial[str(tenFlow.shape)] = tenFlow.new_ones(
            [tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3]])

    # 调整光流以适应网格标准化坐标系
    tenFlow = torch.cat([
        tenFlow[:, 0:1, :, :] * (2.0 / (tenInput.shape[3] - 1.0)),
        tenFlow[:, 1:2, :, :] * (2.0 / (tenInput.shape[2] - 1.0))
    ], 1)

    # 将填充值张量添加到输入图像，以在采样时创建遮罩
    tenInput = torch.cat([tenInput, backwarp_tenPartial[str(tenFlow.shape)]], 1)

    # 使用 grid_sample 进行反向映射，应用光流变换后的位置偏移
    tenOutput = torch.nn.functional.grid_sample(
        input=tenInput,
        grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )

    # 使用掩码去除反向映射中无效的部分
    tenMask = tenOutput[:, -1:, :, :]
    tenMask[tenMask > 0.999] = 1.0
    tenMask[tenMask < 1.0] = 0.0

    # 返回裁剪后的有效图像区域
    return tenOutput[:, :-1, :, :] * tenMask


# end

##########################################################

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # 定义特征提取器模块
        class Extractor(torch.nn.Module):
            def __init__(self):
                super().__init__()

                # 每个级别的特征提取器（共六个级别），将输入图像的特征提取到不同的分辨率
                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                # 继续定义 netThr 到 netSix，以提取不同层次的特征
                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

            def forward(self, tenInput):
                # 将输入逐层通过不同的特征提取模块，得到每层的特征图
                tenOne = self.netOne(tenInput)
                tenTwo = self.netTwo(tenOne)
                tenThr = self.netThr(tenTwo)
                tenFou = self.netFou(tenThr)
                tenFiv = self.netFiv(tenFou)
                tenSix = self.netSix(tenFiv)

                return [tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix]

        # 定义解码器模块，分为多层解码器，每层负责从不同分辨率上恢复光流信息
        class Decoder(torch.nn.Module):
            def __init__(self, intLevel):
                super().__init__()

                # 根据解码级别定义上一层和当前层的输入维度
                intPrevious = \
                [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][
                    intLevel + 1]
                intCurrent = \
                [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][
                    intLevel + 0]

                # 定义特定层级的上采样层、光流平衡因子等
                if intLevel < 6:
                    self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2,
                                                              padding=1)
                    self.netUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32,
                                                              out_channels=2, kernel_size=4, stride=2, padding=1)
                    self.fltBackwarp = [None, None, None, 5.0, 2.5, 1.25, 0.625, None][intLevel + 1]

                # 定义解码器的多层卷积模块
                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3,
                                    stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3,
                                    stride=1, padding=1)
                )

            def forward(self, tenOne, tenTwo, objPrevious):
                tenFlow, tenFeat = None, None

                # 如果是初始层，则仅计算初始特征卷积
                if objPrevious is None:
                    tenVolume = torch.nn.functional.leaky_relu(
                        input=correlation.FunctionCorrelation(tenOne=tenOne, tenTwo=tenTwo),
                        negative_slope=0.1,
                        inplace=False
                    )

                    tenFeat = torch.cat([tenVolume], 1)

                # 否则，进行光流上采样和特征卷积
                else:
                    tenFlow = self.netUpflow(objPrevious['tenFlow'])
                    tenFeat = self.netUpfeat(objPrevious['tenFeat'])

                    tenVolume = torch.nn.functional.leaky_relu(
                        input=correlation.FunctionCorrelation(tenOne=tenOne, tenTwo=backwarp(tenInput=tenTwo,
                                                                                             tenFlow=tenFlow * self.fltBackwarp)),
                        negative_slope=0.1,
                        inplace=False
                    )

                    # 将上采样的光流特征与相关性特征和上一层的特征合并
                    tenFeat = torch.cat([tenVolume, tenOne, tenFlow, tenFeat], 1)

                    # 将合并后的特征通过解码器的每一层卷积模块，逐步提取并生成当前层级的光流
                tenFeat = torch.cat([tenFeat, self.netOne(tenFeat)], 1)
                tenFeat = torch.cat([tenFeat, self.netTwo(tenFeat)], 1)
                tenFeat = torch.cat([tenFeat, self.netThr(tenFeat)], 1)
                tenFeat = torch.cat([tenFeat, self.netFou(tenFeat)], 1)
                tenFeat = torch.cat([tenFeat, self.netFiv(tenFeat)], 1)

                tenFlow = self.netSix(tenFeat)

                # 返回解码器计算的光流和特征图，用于下一层次的解码
                return {'tenFlow': tenFlow, 'tenFeat': tenFeat}

        # 定义细化器，提升光流的精确性
        class Refiner(torch.nn.Module):
            def __init__(self):
                    super().__init__()

                    # 细化器由一系列卷积层组成，进一步处理光流结果
                    self.netMain = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=448 + 2, out_channels=128, kernel_size=3, stride=1, padding=1),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
                    )

            def forward(self, tenInput):
                    # 细化处理输入的光流特征图，生成更精确的光流预测
                    return self.netMain(tenInput)

        self.netExtractor = Extractor()

        self.netTwo = Decoder(2)
        self.netThr = Decoder(3)
        self.netFou = Decoder(4)
        self.netFiv = Decoder(5)
        self.netSix = Decoder(6)

        self.netRefiner = Refiner()

        self.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                              torch.hub.load_state_dict_from_url(
                                  url='http://content.sniklaus.com/github/pytorch-pwc/network-' + args_strModel + '.pytorch',
                                  file_name='pwc-' + args_strModel).items()})
        # end


    def forward(self, tenOne, tenTwo):
        tenOne = self.netExtractor(tenOne)
        tenTwo = self.netExtractor(tenTwo)

        objEstimate = self.netSix(tenOne[-1], tenTwo[-1], None)
        objEstimate = self.netFiv(tenOne[-2], tenTwo[-2], objEstimate)
        objEstimate = self.netFou(tenOne[-3], tenTwo[-3], objEstimate)
        objEstimate = self.netThr(tenOne[-4], tenTwo[-4], objEstimate)
        objEstimate = self.netTwo(tenOne[-5], tenTwo[-5], objEstimate)

        return (objEstimate['tenFlow'] + self.netRefiner(objEstimate['tenFeat'])) * 20.0


# end



# end

netNetwork = None

##########################################################

def estimate(tenOne, tenTwo):
    global netNetwork

    # 检查网络是否已初始化，如果未初始化则进行初始化
    if netNetwork is None:
        netNetwork = Network().cuda().eval()  # 加载网络到GPU并设置为评估模式
    # end

    # 检查输入图像的高度和宽度是否一致
    assert(tenOne.shape[1] == tenTwo.shape[1])
    assert(tenOne.shape[2] == tenTwo.shape[2])

    # 获取输入图像的宽度和高度
    intWidth = tenOne.shape[2]
    intHeight = tenOne.shape[1]

    # 验证输入图像尺寸是否符合要求（如果尺寸不同可注释掉此行以继续）
    assert(intWidth == 1024)  # 如果确认可忽略尺寸限制，则注释此行
    assert(intHeight == 436)  # 如果确认可忽略尺寸限制，则注释此行

    # 将输入图像转换为适合网络的格式，并将其移至GPU
    tenPreprocessedOne = tenOne.cuda().view(1, 3, intHeight, intWidth)
    tenPreprocessedTwo = tenTwo.cuda().view(1, 3, intHeight, intWidth)

    # 计算调整后的宽度和高度，确保尺寸可以被64整除
    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

    # 将输入图像调整为计算后的宽度和高度
    tenPreprocessedOne = torch.nn.functional.interpolate(
        input=tenPreprocessedOne,
        size=(intPreprocessedHeight, intPreprocessedWidth),
        mode='bilinear',
        align_corners=False
    )
    tenPreprocessedTwo = torch.nn.functional.interpolate(
        input=tenPreprocessedTwo,
        size=(intPreprocessedHeight, intPreprocessedWidth),
        mode='bilinear',
        align_corners=False
    )

    # 使用网络估算光流，并将其还原至原始输入图像的尺寸
    tenFlow = torch.nn.functional.interpolate(
        input=netNetwork(tenPreprocessedOne, tenPreprocessedTwo),
        size=(intHeight, intWidth),
        mode='bilinear',
        align_corners=False
    )

    # 还原光流在宽度和高度上的比例，使其与原始尺寸一致
    tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    # 返回光流结果，并将其移动到CPU上
    return tenFlow[0, :, :, :].cpu()

# end

##########################################################

if __name__ == '__main__':
    tenOne = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(args_strOne))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    tenTwo = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(args_strTwo))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

    tenOutput = estimate(tenOne, tenTwo)

    objOutput = open(args_strOut, 'wb')

    numpy.array([ 80, 73, 69, 72 ], numpy.uint8).tofile(objOutput)
    numpy.array([ tenOutput.shape[2], tenOutput.shape[1] ], numpy.int32).tofile(objOutput)
    numpy.array(tenOutput.numpy().transpose(1, 2, 0), numpy.float32).tofile(objOutput)

    objOutput.close()
# end