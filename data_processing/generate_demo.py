#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/13 20:01
# @File    : generate_demo.py
# @Description :

import cv2
import os
from glob import glob

def create_video_from_images(image_folder, output_video_path, fps=30):
    # 获取所有图片路径，并按照文件名排序
    image_files = sorted(glob(os.path.join(image_folder, '*.png')))

    # 确定视频的宽度和高度
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape

    # 初始化视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 设置视频编码
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 遍历所有图片并写入视频
    for image_file in image_files:
        img = cv2.imread(image_file)
        video.write(img)

    # 释放资源
    video.release()
    print(f"视频已保存至 {output_video_path}")

if __name__ == '__main__':
    fps = 15
    image_folder = r"D:\Learning\KEY\CV\code\RIFT-multimodal-image-matching-main\save_image3"  # 图片所在文件夹
    output_video_path = f"D:\Learning\KEY\CV\code\RIFT-multimodal-image-matching-main\output_video_fps{fps}_123.mp4"  # 保存视频的路径


    create_video_from_images(image_folder, output_video_path, fps)
