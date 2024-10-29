#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/19 11:27
# @File    : aedat4tomat.py
# @Description :   补全事件数据，新版本，几种方式
import glob
import os
import numpy as np
import cv2
import dv


# 读取 aedat4文件，提取其中的rgb图像，以及对应时间戳附近的事件数据
def save_frame_event(aedat_file_path=r"D:\Learning\KEY\CV\code\demo\afsf\13\dvSave-2024_10_13_11_21_46.aedat4",
                     output_dir = r"D:\Learning\KEY\CV\code\demo\10_19\13",time_window_us=15000):

    with dv.AedatFile(aedat_file_path) as f:
        # 提取事件数据
        events = np.hstack([packet for packet in f['events'].numpy()])
        timestamps = events['timestamp']
        x_coords = events['x']
        y_coords = events['y']
        polarities = events['polarity']

        # 提取灰度图数据
        if 'frames' in f.names:
            frames = [frame for frame in f['frames']]
        else:
            print("No grayscale frames found in the AEDAT file.")
            return

        # 存储时间窗口内的事件数据
        matched_events = {}

        # 处理每一帧
        for idx, frame in enumerate(frames):
            frame_time = frame.timestamp
            end_time = frame_time
            start_time = end_time - time_window_us

            # 创建掩码，选择在当前时间窗口内的事件
            mask = (timestamps >= start_time) & (timestamps < end_time)
            segment_timestamps = timestamps[mask]
            segment_x_coords = x_coords[mask]
            segment_y_coords = y_coords[mask]
            segment_polarities = polarities[mask]

            # 创建事件体积图
            event_volume = np.zeros((frame.size[1], frame.size[0]), dtype=np.float32)  # 使用帧的尺寸
            for xi, yi in zip(segment_x_coords, segment_y_coords):
                if 0 <= yi < frame.size[1] and 0 <= xi < frame.size[0]:  # 检查坐标是否在范围内
                    event_volume[yi, xi] = 1

            # 保存事件数据
            npz_filename = os.path.join(output_dir, f'events_for_frame_{idx:04d}.npz')
            np.savez(npz_filename, timestamps=segment_timestamps, x_coords=segment_x_coords,
                     y_coords=segment_y_coords, polarities=segment_polarities)

            # 可视化并保存事件的图像表示
            event_image_filename = os.path.join(output_dir, f'events_for_frame_{idx:04d}.png')
            cv2.imwrite(event_image_filename, event_volume * 255)

            # 保存灰度图帧
            grayscale_frame = frame.image  # 假设帧包含 'image' 属性，表示灰度图
            grayscale_filename = os.path.join(output_dir, f'grayscale_frame_{idx:04d}.png')
            cv2.imwrite(grayscale_filename, grayscale_frame)
            print(f"Grayscale frame {idx} saved to {grayscale_filename}")

            print(f"Events saved for frame {idx} to {npz_filename}")

        return matched_events

def event_Completion(base_dir=r'D:\Learning\KEY\CV\code\demo\10_19\13', img_shape=(260, 346)):
    def is_point_in_rect(point, rect_start, rect_end):
        x1, y1 = min(rect_start[0], rect_end[0]), min(rect_start[1], rect_end[1])
        x2, y2 = max(rect_start[0], rect_end[0]), max(rect_start[1], rect_end[1])

        px, py = point
        if x1 <= px <= x2 and y1 <= py <= y2:
            return True
        else:
            return False

    def read_event_npz(path, shape):
        with np.load(path) as data:
            x_data = data['x_coords']
            y_data = data['y_coords']
            # 创建一个全零的矩阵
            event_volume = np.zeros(shape, dtype=np.float32)
            # 将事件数据标记为 1
            for xi, yi in zip(x_data, y_data):
                event_volume[yi, xi] = 1
        return event_volume

    def calculate_homography_and_warp(previous_img, current_img, current_kp, current_des, previous_events, rect_start,
                                      rect_end):
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(previous_img, None)

        # 使用BF匹配器进行特征点匹配
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        knn_matches = bf.knnMatch(des1, current_des, k=2)  # 每个特征点取两个最佳匹配

        # 比率测试来保留优质匹配
        good_matches = []
        for m, n in knn_matches:
            if m.distance < 0.75 * n.distance:  # 如果最优匹配的距离小于次优匹配的75%
                good_matches.append(m)

        # 过滤掉不在指定矩形区域内的特征点
        filtered_matches = []
        for match in good_matches:
            # 获取当前帧中的关键点坐标
            current_point = current_kp[match.trainIdx].pt
            # 检查该点是否在矩形区域内
            if rect_start[0] <= current_point[0] <= rect_end[0] and rect_start[1] <= current_point[1] <= rect_end[1]:
                filtered_matches.append(match)

        if len(filtered_matches) < 4:
            print("匹配点太少，无法计算单应矩阵")
            return None

        # 提取匹配点
        src_pts = np.float32([kp1[m.queryIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([current_kp[m.trainIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)

        # 使用RANSAC估计单应矩阵
        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

        # 绘制匹配点（仅显示在矩形内的点）
        matched_image = cv2.drawMatches(previous_img, kp1, current_img, current_kp, filtered_matches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # 对前一帧事件数据进行透视变换
        warped_events = cv2.warpPerspective(previous_events, matrix,
                                            (previous_events.shape[1], previous_events.shape[0]))
        # warped_events = cv2.warpPerspective(previous_img, matrix,(previous_img.shape[1], previous_img.shape[0]))
        # 显示匹配结果
        # cv2.imshow('Matches', matched_image)
        # cv2.imshow('warped_events', warped_events)
        # cv2.imshow('previous_events', previous_events)
        # cv2.imshow('warped_events+previous_events', warped_events + previous_events)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return warped_events

    def draw_rectangle(event, x, y, flags, param):
        nonlocal rect_start, rect_end, drawing

        if event == cv2.EVENT_LBUTTONDOWN:  # 按下左键，开始绘制矩形
            rect_start = (x, y)
            drawing = True

        elif event == cv2.EVENT_MOUSEMOVE:  # 拖动鼠标，显示矩形
            if drawing:
                rect_end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:  # 松开左键，完成矩形绘制
            rect_end = (x, y)
            drawing = False

    # 红外图像和事件数据路径
    grayscale_path = os.path.join(base_dir, "grayscale_*.png")
    event_path = os.path.join(base_dir, "events_for_frame_*.npz")
    buquan_event_npz = os.path.join(base_dir, "buquan_event_npz")
    buquan_event_png = os.path.join(base_dir, "buquan_event_png")
    os.makedirs(buquan_event_npz, exist_ok=True)
    os.makedirs(buquan_event_png, exist_ok=True)
    grayscale_png = sorted(glob.glob(grayscale_path))
    event_files = sorted(glob.glob(event_path))

    rect_start = None
    rect_end = None
    drawing = False
    best_grayscale_image = None
    best_event_volume = None

    # 从第 5 张红外图像开始处理
    for i in range(4, len(grayscale_png)):
        current_event_volume = read_event_npz(event_files[i], img_shape)
        current_grayscale_image = cv2.imread(grayscale_png[i], cv2.IMREAD_GRAYSCALE)

        if (rect_start is None and rect_end is None) or i % 15 == 0:
            img_copy = current_grayscale_image.copy()
            cv2.namedWindow('Image')
            cv2.setMouseCallback('Image', draw_rectangle)
            temp_img = img_copy.copy()
            # 等待用户完成矩形绘制
            while True:
                # 如果已经开始绘制矩形
                if rect_start is not None and rect_end is not None:
                    cv2.rectangle(temp_img, rect_start, rect_end, (255, 0, 0), 2)
                # 显示图像
                cv2.imshow('Image', temp_img)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # 按下ESC键退出绘制
                    break
            cv2.destroyAllWindows()

        # 计算当前图像的特征点和描述符
        orb = cv2.ORB_create()
        current_kp, current_des = orb.detectAndCompute(current_grayscale_image, None)

        # 初始化当前融合的事件数据
        current_fused_volume = current_event_volume.copy()

        for j in range(1, 4):
            previous_index = i - j  # 前一张图像的索引
            previous_grayscale_image = cv2.imread(grayscale_png[previous_index], cv2.IMREAD_GRAYSCALE)
            previous_event_volume = read_event_npz(event_files[previous_index], img_shape)

            # 计算单应矩阵并进行透视变换
            warped_events = calculate_homography_and_warp(previous_grayscale_image, current_grayscale_image, current_kp,
                                                          current_des, previous_event_volume, rect_start, rect_end)

            # 融合事件数据
            current_fused_volume = np.maximum(current_fused_volume, warped_events)

        if best_event_volume is not None:
            warped_events = calculate_homography_and_warp(best_grayscale_image, current_grayscale_image, current_kp,
                                                          current_des, best_event_volume, rect_start, rect_end)
            current_fused_volume = np.maximum(current_fused_volume, warped_events)

        fused_event_volume = current_fused_volume

        # 保存当前融合后的帧，用于下一帧的补全
        # 每次补全后更新最好的帧为当前帧
        best_event_volume = fused_event_volume
        best_grayscale_image = current_grayscale_image

        # 显示或保存结果
        # cv2.imshow('Fused Event Volume', fused_event_volume)
        cv2.imwrite(os.path.join(buquan_event_png, f'frame_{str(i - 3).zfill(4)}.png'), fused_event_volume * 255)
        y_coords, x_coords = np.nonzero(fused_event_volume)
        # 保存提取到的事件数据为 npz 文件
        np.savez(os.path.join(buquan_event_npz, f'frame_{str(i -3).zfill(4)}.npz'), x_coords=x_coords, y_coords=y_coords)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

if __name__ == '__main__':
    #event_Completion()
    event_Completion()