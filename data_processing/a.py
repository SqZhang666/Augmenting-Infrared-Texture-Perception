#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/27 17:18
# @File    : a.py
# @Description : 文件重命名
import os

# 设置文件夹路径
folder_path = r"D:\Learning\KEY\CV\code\train_data\grayscale\1"  # 将这里替换为你的文件夹路径

# 设置新文件名的起始编号
start_number = 1

# 获取文件夹中所有文件并按名称排序
files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

# 遍历文件并重命名
for file in files:
    # 获取文件的扩展名
    file_ext = os.path.splitext(file)[1]

    # 构造新的文件名，格式化为四位数
    new_name = f"grayscale_frame_{start_number:04d}{file_ext}"

    # 构造旧文件和新文件的完整路径
    old_file_path = os.path.join(folder_path, file)
    new_file_path = os.path.join(folder_path, new_name)

    # 重命名文件
    os.rename(old_file_path, new_file_path)

    # 编号递增
    start_number += 1

print("文件重命名完成！")
