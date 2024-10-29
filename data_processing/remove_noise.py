#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/13 20:00
# @File    : remove_noise.py
# @Description :
import cv2


# 这里直接对补全后的事件数据进行降噪
def remove_noise(base_dir,out_base_dir):
    import numpy as np
    import cv2
    from sklearn.cluster import DBSCAN
    import matplotlib.pyplot as plt
    import os

    def process_file(event_image_path, output_image_path, output_npz_path):
        """处理事件数据图像文件，去除噪点并保存处理结果。"""

        # 读取事件数据图像，假设事件数据为二值化或灰度图像
        event_image = cv2.imread(event_image_path, cv2.IMREAD_GRAYSCALE)

        # 获取所有非零像素（即事件发生的位置）的坐标
        event_points = np.column_stack(np.nonzero(event_image))

        if event_points.shape[0] == 0:
            print(f"文件 {event_image_path} 中没有有效的事件数据。跳过该文件。")
            return

        # 使用 DBSCAN 进行噪点去除
        db = DBSCAN(eps=3, min_samples=4).fit(event_points)
        labels = db.labels_

        # 标记正常点和噪点
        normal_points = event_points[labels != -1]  # 非噪声点

        # 创建去噪后的事件图像，初始化为全零
        denoised_image = np.zeros_like(event_image)

        # 将正常的事件点设置为 255（白色），表示事件发生的像素
        denoised_image[normal_points[:, 0], normal_points[:, 1]] = 255

        # 保存去噪后的图像
        cv2.imwrite(output_image_path, denoised_image)

        # 保存去噪后的事件数据
        processed_x = normal_points[:, 1]
        processed_y = normal_points[:, 0]
        np.savez(output_npz_path, x=processed_x, y=processed_y)

        print(f"去噪后的图像已保存至: {output_image_path}")
        print(f"去噪后的数据已保存至: {output_npz_path}")

    def batch_process_event_images(input_dir, output_dir):
        """批量处理事件数据图像，去除噪点并保存结果。"""
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 批量处理所有事件图像
        for filename in os.listdir(input_dir):
            print(filename)
            if filename.endswith('.png'):  # 假设事件数据是 .png 文件
                file_path = os.path.join(input_dir, filename)
                base_filename = os.path.splitext(filename)[0]  # 去除扩展名
                output_image_path = os.path.join(output_dir, f'clean_{base_filename}.png')
                output_npz_path = os.path.join(output_dir, f'processed_{base_filename}.npz')

                # 处理文件
                print(f"正在处理文件: {file_path}")
                process_file(file_path, output_image_path, output_npz_path)


    output_path = os.path.join(out_base_dir, "cleaned_data1")
    batch_process_event_images(base_dir, output_path)
def remove_noise_npz(base_dir,out_base_dir):
    import numpy as np
    from sklearn.cluster import DBSCAN
    import os

    def process_file(npz_path, output_image_path, output_npz_path):
        # 加载事件数据
        with np.load(npz_path) as data:
            x = data['x_coords']
            y = data['y_coords']
        # 将 x 和 y 坐标点结合成二维数组
        points = np.column_stack((x, y))

        # 使用 DBSCAN 进行噪点去除
        db = DBSCAN(eps=3, min_samples=4).fit(points)
        labels = db.labels_

        # 标记正常点和噪点
        normal_points = points[labels != -1]
        noise_points = points[labels == -1]


        # 保存去噪后的数据
        processed_x = normal_points[:, 0]
        processed_y = normal_points[:, 1]
        np.savez(output_npz_path, x=processed_x, y=processed_y)

        event_volume = np.zeros((260, 346), dtype=np.float32)  # 使用帧的尺寸
        for xi, yi in zip(processed_x, processed_y):
            event_volume[yi, xi] = 1
        cv2.imwrite(output_image_path, event_volume*255)


        print(f"去噪后的数据已保存至: {output_npz_path}")
        print(f"去噪后的图像已保存至: {output_image_path}")

    def batch_process_npz_files(input_dir, output_dir):
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 批量处理所有 npz 文件
        for filename in os.listdir(input_dir):
            if filename.endswith('.npz'):
                file_path = os.path.join(input_dir, filename)
                base_filename = os.path.splitext(filename)[0]  # 去除扩展名
                output_image_path = os.path.join(output_dir, f'clean_{base_filename}.png')
                output_npz_path = os.path.join(output_dir, f'processed_{base_filename}.npz')

                # 处理文件
                print(file_path)
                process_file(file_path, output_image_path, output_npz_path)

    """for root, dirs, files in os.walk(base_dir):
        output_path = os.path.join(out_base_dir, "cleaned_data1")
        for n in dirs:
            dir_path = os.path.join(root, n)
            output_dir_path=os.path.join(output_path, n)
            os.makedirs(output_path, exist_ok=True)
            batch_process_npz_files(dir_path, output_dir_path)"""

    batch_process_npz_files(base_dir, out_base_dir)

if __name__ == '__main__':
    # 对补全后的事件图片进行降噪
    remove_noise_npz(base_dir=r'D:\Learning\KEY\CV\code\demo\10_19\13\buquan_event_npz',out_base_dir=r'D:\Learning\KEY\CV\code\demo\10_19\13\remove_noise')