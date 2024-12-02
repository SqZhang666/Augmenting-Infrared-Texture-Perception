#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/11/7 20:57
# @File    : 1.py
# @Description :
from PIL import Image


def preprocess_ir_image_to_rgb(image_path, target_width=1024, target_height=436, save_path=None):
    # 检查图像是否为三通道 (RGB)
    image = Image.open(image_path)
    if image.mode != 'RGB':
        # 如果图像不是三通道，将单个通道复制到 R、G、B 通道
        image = Image.merge("RGB", (image, image, image))

    # 检查图像尺寸是否为目标尺寸 (1024, 436)
    if image.size != (target_width, target_height):
        # 如果图像尺寸不符合要求，则调整大小
        image = image.resize((target_width, target_height))

    # 如果提供了保存路径，则保存处理后的图像
    if save_path:
        image.save(save_path)

    return image


# 示例：读取、处理和保存两张 IR 图像
image_path_one = r"./rgb/013950.png"
image_path_two = r"./rgb/013955.png"

# 设置保存路径
save_path_one = './images/one.png'
save_path_two = './images/two.png'

# 处理图像并保存
rgb_image_one = preprocess_ir_image_to_rgb(image_path_one, save_path=save_path_one)
rgb_image_two = preprocess_ir_image_to_rgb(image_path_two, save_path=save_path_two)

# 显示处理后的图像
rgb_image_one.show()
rgb_image_two.show()
