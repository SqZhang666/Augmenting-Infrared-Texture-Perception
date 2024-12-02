import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from model import Recursion_Network

from _pix_loss_cls import WeightedLoss

from tqdm import tqdm


class SlidingWindowDataset(Dataset):
    def __init__(self, path, window_size, stride, target_width=1024, target_height=448):
        self.window_size = window_size
        self.stride = stride
        self.target_width = target_width
        self.target_height = target_height
        self.data = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.num_samples = (len(self.data) - self.window_size) // self.stride + 1

        self.transform = transforms.Compose([
            transforms.Resize((self.target_height, self.target_width)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        images = []

        for i in range(start_idx, end_idx):
            try:
                img = Image.open(self.data[i]).convert("RGB") # TODO:网络输入数据需要rgb格式吗
                img = self.transform(img)
                images.append(img)
            except Exception as e:
                raise RuntimeError(f"Error loading image {self.data[i]}: {e}")

        return torch.stack(images)


class Loss(torch.nn.Module):
    def __init__(self, loss_conf_dict):
        super(Loss, self).__init__()
        self.WeightedLoss = WeightedLoss(loss_conf_dict=loss_conf_dict)

    def forward(self, output, target):
        return self.WeightedLoss(output, target)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 数据集路径
    data_path = "train_data"

    frame_number = 5

    # 创建数据集和 DataLoader
    dataset = SlidingWindowDataset(data_path, window_size=frame_number, stride=1, target_width=1024, target_height=448)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = Recursion_Network(number=frame_number).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss_conf_dict = {'CharbonnierLoss': 1.0, 'SSIMLoss': 0.05, 'EdgeLoss': 0.05}
    loss_fun = Loss(loss_conf_dict)

    for epoch in range(10):  # 假设10个epoch
        model.train()  # 训练模式
        print(f"Epoch {epoch + 1} started")

        epoch_loss = 0  # 累积当前 epoch 的损失
        id = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}", unit="batch"):
            optimizer.zero_grad()

            input = batch.to(device)
            outputs = model(input)

            # 计算损失
            loss_values = loss_fun(outputs, input[0, -1, ...,].unsqueeze(0))
            loss_values.backward()
            optimizer.step()

            epoch_loss += loss_values.item()  # 累加当前 batch 的损失

            # 保存模型输出图像
            output_image = outputs[0].clamp(0, 1)  # 保证输出在 [0, 1] 范围内
            output_image = transforms.ToPILImage()(output_image)  # 转换为 PIL 图像
            outpath = os.path.join('outdata', f"output_epoch_{epoch + 1}_batch_{id}.png")
            id += 1
            output_image.save(outpath)  # 保存图像

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader)}")


