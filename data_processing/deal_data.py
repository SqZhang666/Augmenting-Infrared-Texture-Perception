#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/14 19:25
# @File    : deal_data.py
# @Description :
# 1.批量清理一定区域内的事件数据 clean_data()
# 2.清理，加工补全后的事件数据 match_buquanevent_deal_grayscale() 使用二值化图片限定事件数据
# 3.eroded_image 腐蚀产生二值化图片
# 4.new_event 利用二值化图片变换后得到的整体mask，进一步得到变换后的仅有event的图像

import numpy as np
import cv2
import os

def light_and_contrast():
    from PIL import Image, ImageEnhance
    import os

    # 批量处理图片的文件夹路径
    input_folder = r'D:\Learning\KEY\CV\code\demo\10_19\13\grayscale'  # 输入图片的文件夹
    output_folder = r'D:\Learning\KEY\CV\code\demo\10_19\13\deal_grayscale'  # 输出处理后的图片文件夹

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 调整参数
    brightness_factor = 0.5  # 亮度调整系数，<1 减少亮度
    contrast_factor = 3    # 对比度调整系数，>1 增加对比度
    threshold_value = 128    # 黑白阈值(0-255)，128 为中间值

    # 遍历输入文件夹中的所有图片
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # 仅处理jpg和png图片
            # 打开图像
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)

            # 调整亮度
            enhancer_brightness = ImageEnhance.Brightness(img)
            img_brightness = enhancer_brightness.enhance(brightness_factor)

            # 调整对比度
            enhancer_contrast = ImageEnhance.Contrast(img_brightness)
            img_contrasted = enhancer_contrast.enhance(contrast_factor)

            # 转为灰度图
            img_gray = img_contrasted.convert('L')

            # 二值化，使用阈值
            img_bw = img_gray.point(lambda x: 255 if x > threshold_value else 0, '1')

            # 保存处理后的图像到输出文件夹
            output_path = os.path.join(output_folder, filename)
            img_bw .save(output_path)

    print("批量处理完成！")

def clean_data():
    # 读取图片并生成事件矩阵
    def read_event_image(path):
        event_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return event_image

    # 清理事件图片
    def clean_event_image(path, file, output_dir, xmin,xmax,ymin,ymax):
        # 加载图片
        event_image = read_event_image(os.path.join(path, file))

        # 将 xmin 以上的事件点清理掉
        event_image[:, xmax:] = 255
        event_image[:, :xmin] = 255
        event_image[ymax:, :] = 255
        event_image[:ymin, :] = 255

        # 保存清理后的图片
        image_output_path = os.path.join(output_dir, f"{file}")
        cv2.imwrite(image_output_path, event_image)
        print(f"清理完成并保存图像 {image_output_path}")

    # 批量处理目录中的所有图片文件
    def batch_process_event_files(input_dir, output_dir, xmin,xmax,ymin,ymax):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 遍历输入目录中的所有图片文件
        for file in os.listdir(input_dir):
            if file.endswith('.png') or file.endswith('.jpg'):  # 根据实际图片格式调整
                print(f"正在处理文件: {file}")
                clean_event_image(input_dir, file, output_dir, xmin,xmax,ymin,ymax)

    # 设置路径和参数
    input_dir = r"D:\Learning\KEY\CV\code\demo\10_19\13\deal_grayscale_after"
    output_dir = r"D:\Learning\KEY\CV\code\demo\10_19\13\a1"
    xmin,xmax = 10,140
    ymin,ymax = 50,160

    # 执行批量处理
    batch_process_event_files(input_dir, output_dir, xmin,xmax,ymin,ymax)

def match_buquanevent_deal_grayscale():
    def fuc():
        buquan_event_path = r'D:\Learning\KEY\CV\code\demo\10_19\13\buquan_event_png'
        deal_grayscale_path = r'D:\Learning\KEY\CV\code\demo\10_19\13\a1'

        buquan_event_files = sorted(os.listdir(buquan_event_path))
        deal_grayscale_files = sorted(os.listdir(deal_grayscale_path))

        for event_file ,grayscale_file in zip(buquan_event_files, deal_grayscale_files):
            event_img = cv2.imread(os.path.join(buquan_event_path, event_file), cv2.IMREAD_GRAYSCALE)
            grayscale_img = cv2.imread(os.path.join(deal_grayscale_path, grayscale_file), cv2.IMREAD_GRAYSCALE)

            #cv2.imshow('event_img', event_img)
            binary_img = ((255 - grayscale_img) > 0).astype(np.uint8)
            binary_img[binary_img > 0] = 1
            #cv2.imshow('grayscale_img', binary_img*255)

            deal_event_img = event_img * binary_img

            #cv2.imshow('deal_event_img', deal_event_img)
            #cv2.waitKey(0)
            path = r'D:\Learning\KEY\CV\code\demo\10_19\13\deal_event_img1'
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(os.path.join(path,event_file), deal_event_img)

    fuc()

def eroded_image():
    # 输入和输出文件夹路径
    input_folder = r'D:\Learning\KEY\CV\code\demo\10_19\13\deal_grayscale'  # 替换为你的输入图片文件夹路径
    output_folder = r'D:\Learning\KEY\CV\code\demo\10_19\13\deal_grayscale_after'  # 替换为你想要保存的输出文件夹路径
    comparison_folder = r'D:\Learning\KEY\CV\code\demo\10_19\13\deal_grayscale_after1'  # 用于保存对比图的文件夹路径

    # 创建输出和对比文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(comparison_folder, exist_ok=True)

    # 定义腐蚀核（可以调整尺寸以改变缩小程度）
    kernel = np.ones((11, 11), np.uint8)

    # 遍历输入文件夹中的所有图片文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.png') or filename.endswith('.jpg'):  # 可根据图片格式调整
            # 构建完整路径
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            comparison_path = os.path.join(comparison_folder, f"comparison_{filename}")

            # 读取图片并转为灰度图
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: Failed to read {image_path}")
                continue

            # 二值化处理，突出黑色方框
            _, binary_image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY_INV)

            # 执行腐蚀操作，缩小方框边缘
            eroded_image = cv2.erode(binary_image, kernel, iterations=1)

            # 将腐蚀后的图像反转，恢复为原始风格
            final_image = cv2.bitwise_not(eroded_image)

            # 保存处理后的图片到输出文件夹
            cv2.imwrite(output_path, final_image)

            # 使用像素相减的方式进行对比
            difference_image = cv2.absdiff(image, final_image)

            # 为了更清晰地看到差异区域，可以进行增强显示
            _, enhanced_difference = cv2.threshold(difference_image, 30, 255, cv2.THRESH_BINARY)

            # 保存直接相减的对比图像
            cv2.imwrite(comparison_path, enhanced_difference)
            print(f"Processed and saved: {comparison_path}")

    print("Batch processing completed.")




if __name__ == '__main__':
    """# 先腐蚀产生缩小后的二值图
    eroded_image()
    # 清理二值图，只留下纹理部分
    clean_data()
    # 利用纹理部分，约束产生对应的事件数据
    match_buquanevent_deal_grayscale()"""










