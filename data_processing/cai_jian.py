import os
from PIL import Image

def crop_and_save_images(source_dir, target_dir, crop_region):
    """
    裁剪指定目录下的所有图片并保存到目标目录。

    Args:
        source_dir (str): 源图片文件夹路径。
        target_dir (str): 裁剪后图片保存的目标文件夹路径。
        crop_region (tuple): 裁剪区域，格式为 (x, y, width, height)。
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 获取裁剪区域
    x, y, width, height = crop_region

    # 遍历源文件夹中的所有图片
    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)
        # 检查文件是否为图片
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            try:
                with Image.open(file_path) as img:
                    # 裁剪图片
                    cropped_img = img.crop((x, y, x + width, y + height))
                    # 保存裁剪后的图片
                    save_path = os.path.join(target_dir, filename)
                    cropped_img.save(save_path)
                    print(f"Saved cropped image: {save_path}")
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

if __name__ == "__main__":
    # 指定源目录和目标目录
    source_dir = r"D:\Learning\KEY\CV\code\Augmenting_Infrared_Texture_Perception\train\train_data"  # 替换为实际路径
    target_dir = r"C:\Users\zsq\Desktop\a"

    # 指定裁剪区域 (x, y, width, height)
    crop_region = (140, 140, 204, 204)  # 替换为实际裁剪区域

    crop_and_save_images(source_dir, target_dir, crop_region)
