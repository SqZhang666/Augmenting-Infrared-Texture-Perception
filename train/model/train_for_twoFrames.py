import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Network import Network
from _pix_loss_cls import WeightedLoss
from tqdm import tqdm
import argparse
from torch.optim.lr_scheduler import StepLR


# Define the Loss class
class Loss(torch.nn.Module):
    def __init__(self, number):
        from PWCNet import PWCNet
        super(Loss, self).__init__()
        self.number = number
        self.mse = torch.nn.MSELoss()
        self.PWCNet = PWCNet().cuda()
        # self.diagonal = (112 ** 2 + 256 ** 2) ** 0.5

    def forward(self, output, target, target_flows):
        flow_loss = 0

        for i in range(self.number - 1):
            with torch.no_grad():
                pre_flow = self.PWCNet(target[i + 1, ...].unsqueeze(0), output).detach()
            print(pre_flow.max())
            flow_loss += self.mse(target_flows[i], pre_flow)
        loss_value1 = self.mse(output, target[0, ...].unsqueeze(0))
        loss_value2 = flow_loss / self.number
        return 0.5 * loss_value1 + 0 * loss_value2


# Data reading function
def read_data(path, device):
    transform = transforms.Compose([
        transforms.Resize((448, 1024)),
        transforms.ToTensor()
    ])
    frames = []
    image_files = sorted([f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    for image_file in image_files:
        image_path = os.path.join(path, image_file)
        image = Image.open(image_path).convert("RGB")
        frames.append(transform(image).to(device))
    frames_tensor = torch.stack(frames)
    return frames_tensor


# Training function
def train_model(model, frames, frame_irs, loss_fun, optimizer, scheduler, epochs, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)
    train_loss = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs, target_flows = model(frames, frame_irs)
        loss_values = loss_fun(outputs, frame_irs, target_flows)
        loss_values.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        train_loss.append(loss_values.item())

        if epoch % (epochs // 20) == 0 or epoch == epochs - 1:
            with torch.no_grad():
                model.eval()
                train_outputs, _ = model(frames, frame_irs)
                train_output_image = transforms.ToPILImage()(train_outputs[0])
                train_output_image.save(os.path.join(save_dir, f"train_epoch{epoch}.png"))
        torch.cuda.empty_cache()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss_values.item():.4f}")

    return train_loss


# Plot loss curve
def plot_loss(train_loss, save_path):
    plt.figure()
    plt.title('Loss during training')
    plt.plot(range(len(train_loss)), train_loss, label="Train Loss")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()


# Main function
def main(args):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frames = read_data(args.rgb_path, device)
    frame_irs = read_data(args.ir_path, device)

    model = Network(number=args.frame_number).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    loss_fun = Loss(args.frame_number)

    train_loss = train_model(
        model,
        frames,
        frame_irs,
        loss_fun,
        optimizer,
        scheduler,
        args.epochs,
        args.output_dir,
        device
    )

    plot_loss(train_loss, os.path.join(args.output_dir, 'loss.png'))

    save_path = os.path.join(args.output_dir, 'model_checkpoint.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss
    }, save_path)

    print(f"Model and training artifacts saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the Network Model")
    parser.add_argument('--rgb_path', type=str, default=r'./rgb', help="Path to RGB frames")
    parser.add_argument('--ir_path', type=str, default=r'./ir', help="Path to IR frames")
    parser.add_argument('--frame_number', type=int, default=3, help="Number of input frames")
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--lr_step_size', type=int, default=10, help="Step size for learning rate scheduler")
    parser.add_argument('--lr_gamma', type=float, default=0.5, help="Gamma for learning rate scheduler")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--output_dir', type=str, default='./outdata', help="Directory to save outputs")

    args = parser.parse_args()
    main(args)
