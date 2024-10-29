#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/9/26 09:13
# @File    : New4_step.py
# @Description : 对应All_model3，事件数据没有取反，事件点还是白色
#                   不再进行分块处理
#               使用新数据 12 号文件夹，里边只有 train_data3
#TODO:  1.判断模型训练结束的时机，而不是指定训练多少轮次
#       2.模型结构设计有什么要求，合理性？那里需要变到一维上吗?大小有要求吗？

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from All_model4 import model_test
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset, Dataset



# 定义MSE损失函数
class MSELossFunction(nn.Module):
    def __init__(self):
        super(MSELossFunction, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        loss = self.mse(output, target)
        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.criterion = nn.MSELoss()  # 或者根据需求使用其他损失函数
        # self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, y_true, y_pred):
        return self.criterion(y_pred, y_true)


class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(CombinedLoss, self).__init__()
        self.edge_loss = EdgeLoss()
        self.mse_loss = MSELossFunction()
        self.alpha = alpha  # 权重alpha
        self.beta = beta  # 权重beta

    def forward(self, y_true, y_pred, output, target):
        # 计算每个损失
        loss1 = self.edge_loss(y_true, y_pred)
        loss2 = self.mse_loss(output, target)
        # 返回加权总损失
        return self.alpha * loss1 + self.beta * loss2


# 定义可视化函数
def visualize(epoch,index, output, num_blocks_h, num_blocks_w, block_height, block_width):
    output = (output * 255).astype(np.uint8)  # 还原到 [0, 255] 范围并转换为 uint8
    image = np.zeros((num_blocks_h * block_height, num_blocks_w * block_width), dtype=np.uint8)
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            image[i * block_height:(i + 1) * block_height, j * block_width:(j + 1) * block_width] = output[
                i * num_blocks_w + j]
    plt.imshow(image, cmap='gray')
    plt.title(f'Epoch {epoch + 1}')
    plt.axis('off')
    save_path = os.path.join('results', f'epoch4_{epoch + 1}_{index}.png')
    plt.savefig(save_path)
    plt.close()


# 保存损失到文本文件
def save_loss_to_txt(loss, filename):
    with open(filename, 'a') as f:
        f.write(f'{loss}\n')


# 绘制损失曲线
def plot_loss_curve(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    losses = [float(line.strip()) for line in lines]
    plt.figure()
    plt.plot(losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig('results/loss_curve.png')
    plt.show()


def load_event_data(file_paths):
    event_volumes = []
    for file_path in file_paths:
        with np.load(file_path) as data:
            x = data['x']
            y = data['y']
            # 创建一个全零的矩阵
            event_volume = np.zeros((1280, 704), dtype=np.float32)
            # 将事件数据标记为 1
            for xi, yi in zip(x, y):
                if 0 <= xi < 704 and 0 <= yi < 1280:
                    event_volume[yi, xi] = 1
        event_volumes.append(event_volume)
    # 堆叠成一个 3D 体积
    event_volume_3d = np.stack(event_volumes, axis=0)
    return event_volume_3d


def load_and_edge_detect_nir(nir_image_path):
    nir_image = cv2.imread(nir_image_path, cv2.IMREAD_GRAYSCALE)
    # nir_image = cv2.medianBlur(nir_image, 1)
    # nir_image = cv2.bilateralFilter(nir_image, 5, 75, 75)

    edges = cv2.Canny(nir_image, 10, 100)
    kernel = np.ones((1, 1), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    original_height, original_width = edges.shape

    # 计算所需的填充量
    target_width, target_height = (704, 1280)

    right_padding = max(0, target_width - original_width)
    bottom_padding = max(0, target_height - original_height)

    # 使用 cv2.copyMakeBorder 进行填充
    # noinspection PyTypeChecker
    edges_padded = cv2.copyMakeBorder(
        edges,
        top=0,  # 上部不填充
        bottom=bottom_padding,  # 只填充底部
        left=0,  # 左侧不填充
        right=right_padding,  # 只填充右侧
        borderType=cv2.BORDER_CONSTANT,
        value=0  # 使用黑色（0）进行填充
    )

    # 保存处理后的图像
    out_path = r'data/raw_edges_resize.png'
    cv2.imwrite(out_path, edges_padded)

    return edges_padded

class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        self._prepare_dataset()

    def _prepare_dataset(self):
        for group_folder in sorted(os.listdir(self.root_dir)):
            group_folder_path = os.path.join(self.root_dir, group_folder)
            if not os.path.isdir(group_folder_path):
                continue

            train_data_path = os.path.join(group_folder_path, 'train_data4')
            train_data_files = sorted(os.listdir(train_data_path))

            for i in range(len(train_data_files)):
                self.samples.append({
                    'train_data':os.path.join(train_data_path, train_data_files[i]),
                })

    def __getitem__(self, index):
        #加载路径
        sample_path = self.samples[index]
        train_data = np.load(sample_path['train_data'])
        return train_data,int(sample_path['train_data'][-7:-4])
    def __len__(self):
        return len(self.samples)


# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = model_test(in_channels=1).to(device)
loss_fn = CombinedLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_filename = 'results/loss_values.txt'



if __name__ == '__main__':
    current_dir = os.path.dirname(__file__)

    if not os.path.exists('results'):
        os.makedirs('results')

    # 训练循环
    model.train()
    num_epochs = 200
    batch_size = 1

    # 定义数据路径
    data_root_dir = r'train_data/'
    datasets = CustomDataset(data_root_dir)
    train_loader = DataLoader(dataset=datasets, batch_size=batch_size, shuffle=False, num_workers=0)

    for epoch in tqdm(range(num_epochs), desc='Processing', ncols=80, position=0, leave=True):
        epoch_loss = 0.0
        for i, X in enumerate(train_loader):
            data,num = X[0],X[1]
            v_inputs =data[:,0,...].float()
            input_batch = v_inputs[:, 0:6, ...].to(device)
            target_batch = v_inputs[:, 6:7, ...].to(device)
            y_true_batch = v_inputs[:, 7:8, ...].to(device)
            mask_batch = v_inputs[:, 8:9, ...].to(device)

            optimizer.zero_grad()
            event_data, output = model(input_batch)

            #loss = loss_fn((input_batch[:,3:4,...].to(torch.int)|mask_batch.to(torch.int)).to(torch.float), event_data, output, target_batch)
            loss = loss_fn(mask_batch*y_true_batch,event_data*mask_batch,output,target_batch)
            #loss = loss_fn(input_batch[:,3:4,...],event_data*mask_batch,output,target_batch)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

            if epoch % 25 == 0 and i % 5 == 0:
                print(f'正在进行第{epoch+1}轮次，第{i}组数据的测试推理')
                v_inputs_test =data[:,0,...].float()
                with torch.no_grad():
                    input_batch_test = v_inputs_test[:, 0:6, ...].to(device)
                    event_data, output = model(input_batch_test)

                    event_img = (event_data.detach().cpu().numpy()[0,0,:,:]  * 255).astype(np.uint8)
                    output_image = (output.detach().cpu().numpy()[0,0,:,:] * 255).astype(np.uint8)

                    plt.figure(figsize=(10,5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(event_img, cmap='gray')
                    plt.title('event_img')

                    plt.subplot(1, 2, 2)
                    plt.imshow(output_image, cmap='gray')
                    plt.title('output')

                    plt.axis('off')
                    save_path = os.path.join('results', f'output5_{epoch + 1}_{num}.png')
                    plt.savefig(save_path)
                    plt.close()


        epoch_loss /= i+1
        save_loss_to_txt(epoch_loss, loss_filename)
        # 合并所有输出并处理重叠区域
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'results/new_my_model_weights_epoch5.pth')
    print("Training complete.")
    plot_loss_curve(loss_filename)
