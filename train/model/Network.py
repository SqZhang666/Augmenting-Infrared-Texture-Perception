import torch
import torch.nn as nn
from modules import Conv, ResBlock, MLP, CUnet, Deconv
from PWCNet import PWCNet
from utils import PositionalEncoding


class Network(nn.Module):
    def __init__(self, number):
        super(Network, self).__init__()
        self.number = number
        self.time_idx = torch.linspace(0, 1, number).unsqueeze(0).t().to('cuda')  # time idx

        # params
        n_resblock = 4
        n_feats = 32
        kernel_size = 3
        padding = 1
        n_colors = 3

        pos_b, pos_l = 1.25, 80  # position encoding params
        mlp_dim_list = [2 * pos_l, 512, n_feats * 4 * 2]  # (160, 512, 256)
        # mlp_dim_list = [2 * pos_l, 336, n_feats * 4 * 2 * 2] # (160, 336, 512)
        mlp_act = 'gelu'

        self.PWCNet = PWCNet()

        OPTF_module = [
            Conv(input_channels=2, n_feats=n_feats, kernel_size=kernel_size, padding=padding, act=True),
            ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
            ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
            ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding)]
        self.OPTF = nn.Sequential(*OPTF_module)

        self.INRF = CUnet(n_feats=n_feats, n_resblock=n_resblock, kernel_size=kernel_size, padding=padding)

        FeatureBlock = [
            Conv(input_channels=n_colors, n_feats=n_feats, kernel_size=kernel_size, padding=padding, act=True),
            ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
            ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
            ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding)]
        self.feature = nn.Sequential(*FeatureBlock)

        self.pe_t = PositionalEncoding(pe_embed_b=pos_b, pe_embed_l=pos_l)

        self.embed_mlp = MLP(dim_list=mlp_dim_list, act=mlp_act)

        Fusion_moudle = [
            Conv(input_channels=n_feats * 2, n_feats=n_feats, kernel_size=kernel_size, padding=padding, act=True),
            ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
            ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
            ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding)]
        self.Fusion = nn.Sequential(*Fusion_moudle)

        OutBlock = [
            ResBlock(Conv, n_feats=(self.number - 1) * n_feats, kernel_size=kernel_size, padding=padding),
            ResBlock(Conv, n_feats=(self.number - 1) * n_feats, kernel_size=kernel_size, padding=padding),
            Deconv(input_channels=(self.number - 1) * n_feats, n_feats=n_colors, kernel_size=kernel_size, stride=1,
                   padding=padding)]
        self.out = nn.Sequential(*OutBlock)

        self.final_layer = nn.Sigmoid()

    def forward(self, frames, frame_irs):
        # n - 1 个模块
        # frames: number个，分别计算到当前帧的光流，共 n-1 个flow
        # frames：索引0号是当前帧
        # frane_irs: number个，用于结合 n-1 个flow，输入fusion得到四个fusion_feature,最终解码得到预测的当前红外帧

        t_pe_ = [self.pe_t(idx) for idx in self.time_idx]  # [frame_num*[pos_l*2,1]]
        t_pe = torch.cat(t_pe_, dim=0)  # [frame_num, pos_l*2]
        t_embed = self.embed_mlp(t_pe)  # [frame_num, n_feats*4*2]

        frame_t = frames[0,].unsqueeze(0)
        # feature = self.feature(frame_t)  # [1, c, h, w]

        # print('frame_t:', frame_t.shape)
        # print('feature_ir feature:', feature_ir.shape)
        output_fusion_list = []
        target_flows = []
        for n in range(len(self.time_idx) - 1):
            frame_n = frames[n + 1, :, :, :].unsqueeze(0)
            flow_n = self.PWCNet(frame_n, frame_t)  # [B, 2, H, W] #TODO：这里算出来的光流是谁到谁

            target_flows.append(flow_n)
            flow_n = torch.nn.functional.interpolate(input=flow_n, size=(448, 1024), mode='bilinear',
                                                     align_corners=False)  # 1,2,h,w
            flow_feature = self.OPTF(flow_n)  # 1,c,h,w
            output_inrf = self.INRF(flow_feature, t_embed[n + 1])  # [B, C, H, W]
            # print(f"output_inrf shape: {output_inrf.shape}") #[1, 32, 128, 128])

            frame_ir = frame_irs[n + 1, :, :, :].unsqueeze(0)
            feature_ir = self.feature(frame_ir)  # [1, c, h, w]

            output_fusion = self.Fusion(torch.cat([feature_ir, output_inrf], dim=1))  # 1,c,h,w
            # print(f"Fusion output shape: {output_fusion.shape}") #[1, 32, 128, 128])
            output_fusion_list.append(output_fusion)

        fusion_result = torch.cat(output_fusion_list, dim=1)
        # print(f"fusion_result shape: {fusion_result.shape}")
        output_image = self.out(fusion_result)

        return self.final_layer(output_image), target_flows


# Testing the network
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 1
    height, width = 128, 128
    frame_num = 2

    frames = []
    for _ in range(frame_num):
        frames.append(torch.randn(batch_size, 3, height, width).to(device))
    frames = torch.stack(frames)

    model = Network(number=frame_num).to(device)
    model.eval()  # Set model to evaluation mode

    # print(frames)
    output_image = model(frames)

    print(f"Final output image shape: {output_image.shape}")  # 1,3,128,128
