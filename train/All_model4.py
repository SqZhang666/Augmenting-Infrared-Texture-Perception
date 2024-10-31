#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/28 18:27
# @File    : train.py
# @Description : V0.1
# 采用实际采集的数据集进行测试训练
# 相较于之前的New4_step系列训练文件，修正了数据集的加载方式，另外对于loss进行了新的约束
#TODO 1.对于loss那里我已经得到了mask，target是否有存在的必要
#     2.纹理部分是否可以直接给出，就是其实不需要融合后的图片，毕竟求解loss的时候也是分开求解的
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from All_model5 import model_test
from torch.utils.data import DataLoader, Dataset
from soberloss import GradLoss



class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(CombinedLoss, self).__init__()
        self.texture_loss = nn.MSELoss() # 纹理部分约束
        self.subject_loss = nn.MSELoss() # 排除纹理的其他约束
        self.alpha = alpha  # 纹理部分权重alpha
        self.beta = beta  # 除纹理的其他部分权重beta

    def forward(self,output, target, mask):
        # 计算每个损失
        binary_mask = (mask >= 1).to(torch.int)
        loss1 = self.texture_loss(output * binary_mask, target*binary_mask) # 只关注纹理部分
        loss2 = self.subject_loss(output * (1-binary_mask), target*(1-binary_mask)) # 纹理之外的主体部分
        # 返回加权总损失
        loss = self.alpha*loss1 + self.beta*loss2 
        return self.texture_loss(output,target)


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

            # 忽略隐藏文件夹以及非目录
            if group_folder.startswith('.') or not os.path.isdir(group_folder_path):
                continue

            # 定义每种文件的路径
            ir_path = os.path.join(group_folder_path, 'ir')
            new_event_path = os.path.join(group_folder_path, 'new_event')
            grayscale_path = os.path.join(group_folder_path, 'grayscale')
            target_path = os.path.join(group_folder_path, 'target_fusion')
            mask_path = os.path.join(group_folder_path, 'mask')

            # 获取每种文件夹的文件列表，并确保它们的数量一致
            grayscale_files = sorted(os.listdir(grayscale_path))
            ir_files = sorted(os.listdir(ir_path))
            new_event_files = sorted(os.listdir(new_event_path))
            target_files = sorted(os.listdir(target_path))
            mask_files = sorted(os.listdir(mask_path))

            # 确保所有文件夹中的文件数量一致
            num_files = len(grayscale_files)
            if not (num_files == len(ir_files) == len(new_event_files) == len(target_files) == len(mask_files)):
                print(num_files,len(ir_files),len(new_event_files),len(target_files),len(mask_files))
                raise ValueError(f"文件数量不一致: {group_folder}")

            # 使用 zip 将所有文件路径一一对应
            for ir, new_event, target_fusion, mask in zip(
                ir_files, new_event_files, target_files, mask_files
            ):
                self.samples.append({
                    'ir': os.path.join(ir_path, ir),
                    'new_event': os.path.join(new_event_path, new_event),
                    'target_fusion': os.path.join(target_path, target_fusion),
                    'mask': os.path.join(mask_path, mask)
                })

            #'grayscale': cv2.imread(sample_path['grayscale'], cv2.IMREAD_GRAYSCALE),
    def __getitem__(self, index):
        sample_path = self.samples[index]
        #print(sample_path)
        train_data = {
            'ir': cv2.imread(sample_path['ir'], cv2.IMREAD_GRAYSCALE),
            'new_event': cv2.imread(sample_path['new_event'], cv2.IMREAD_GRAYSCALE),
            'target_fusion': cv2.imread(sample_path['target_fusion'], cv2.IMREAD_GRAYSCALE),
            'mask': cv2.imread(sample_path['mask'],cv2.IMREAD_GRAYSCALE)
        }
        return train_data

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
        for i, data in enumerate(train_loader):
            new_event = cv2.resize(data['new_event'][0,:,:].numpy(), (346, 260))
            ir = cv2.resize(data['ir'][0,:,:].numpy(), (346, 260))
            target = cv2.resize(data['target_fusion'][0,:,:].numpy(), (346, 260))
            mask = cv2.resize(data['mask'][0,:,:].numpy(), (346, 260))
            
            #binary_mask = (mask >= 255).astype(int)
            
            
            train_ir = torch.from_numpy(np.expand_dims(np.expand_dims(ir, axis=0), axis=0) / 255.0).float()
            train_event = torch.from_numpy(np.expand_dims(np.expand_dims(new_event, axis=0), axis=0) / 255.0).float()
            #train_event = torch.from_numpy(np.expand_dims(np.expand_dims(target*binary_mask, axis=0), axis=0) / 255.0).float()
            train_target = torch.from_numpy(np.expand_dims(np.expand_dims(target, axis=0), axis=0) / 255.0).float()
            train_mask = torch.from_numpy(np.expand_dims(np.expand_dims(mask, axis=0), axis=0) / 255.0).float()
            train_ir, train_event, train_target, train_mask = train_ir.to(device), train_event.to(device), train_target.to(device),train_mask.to(device)

            optimizer.zero_grad()
            output = model(train_ir,train_event)



            #loss = loss_fn(output, train_target,train_mask)
            loss = loss_fn(output, train_target,train_mask)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

            if epoch % 25 == 0 and i % 5 == 0:
                print(f'正在进行第{epoch+1}轮次，第{i}组数据的测试推理')
                with torch.no_grad():
                    pre_output = model(train_ir,train_event)

                    target_img = (train_target.detach().cpu().numpy()[0,0,:,:] * 255).astype(np.uint8)
                    output_image = (pre_output.detach().cpu().numpy()[0,0,:,:] * 255).astype(np.uint8)

                    plt.figure(figsize=(10,5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(target_img, cmap='gray')
                    plt.title('target_img')

                    plt.subplot(1, 2, 2)
                    plt.imshow(output_image, cmap='gray')
                    plt.title('output_image')

                    plt.axis('off')
                    save_path = os.path.join('results', f'5output5_{epoch + 1}_{i}.png')
                    plt.savefig(save_path)
                    plt.close()


        epoch_loss /= i+1
        save_loss_to_txt(epoch_loss, loss_filename)
        # 合并所有输出并处理重叠区域
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.6f}')

    # 保存模型
    torch.save(model.state_dict(), 'results/model5_1028.pth')
    print("Training complete.")
    plot_loss_curve(loss_filename)
