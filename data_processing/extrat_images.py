import numpy as np
import cv2
import os
import dv
import glob
def event_nir_fusion(base_dir, fps=50, time_window_ms=15):

    def time_difference(time1, time2):
        # 解析时分秒
        h1, m1, s1 = int(time1[:2]), int(time1[2:4]), int(time1[4:])
        h2, m2, s2 = int(time2[:2]), int(time2[2:4]), int(time2[4:])

        # 将时分秒转换为总秒数
        total_seconds1 = h1 * 3600 + m1 * 60 + s1
        total_seconds2 = h2 * 3600 + m2 * 60 + s2

        # 计算两个时间的差值
        diff_seconds = abs(total_seconds1 - total_seconds2)

        return diff_seconds

    def process_raw_video(file_path, fps, output_dir):
        # 打开原始数据文件
        with open(file_path, 'rb') as f:
            raw_data = f.read()

        file_name, _ = os.path.splitext(os.path.basename(file_path))
        nir_start_time = file_name[-9:-3]
        print(nir_start_time)

        # 目标帧尺寸
        frame_height, frame_width, num_channels = 512, 640, 2
        frame_size = frame_height * frame_width * num_channels

        # 计算帧数
        num_frames = len(raw_data) // frame_size
        print(f"Total number of frames: {num_frames}")

        # 加载原始数据到 NumPy 数组，并确保其形状
        raw_data_array = np.frombuffer(raw_data, dtype=np.uint8)
        assert raw_data_array.size == num_frames * frame_size, "数据大小不匹配！"

        """# 创建 RGB 图像数据数组
        cvt_data = np.zeros([num_frames, frame_height, frame_width, 3], dtype=np.uint8)"""

        # 计算每帧的时间间隔
        time_interval = 1 / fps  # 每帧时间间隔（秒）

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 存储帧的时间戳
        frame_timestamps = []

        # 处理每一帧
        for idx in range(num_frames):
            """# 转换 YUV422 到 RGB
            cvt_data[idx, ...] = cv2.cvtColor(final_data[idx, ...], cv2.COLOR_YUV2RGB_Y422)"""

            # 提取当前帧的 YUV422 数据
            start_idx = idx * frame_size
            end_idx = start_idx + frame_size
            frame_data = raw_data_array[start_idx:end_idx]

            # 重塑为 (height, width, 2) 以匹配 YUV422 格式
            yuv_frame = frame_data.reshape((frame_height, frame_width, 2))

            # 只提取 Y 通道，Y 通道是每个像素的亮度值
            y_channel = yuv_frame[:, :, 0]

            # 计算时间戳
            timestamp = idx * time_interval
            frame_timestamps.append(timestamp)

            """# 显示图像
            gray_data = cv2.cvtColor(cvt_data[idx, ...][:256, :320], cv2.COLOR_RGB2GRAY)"""

            cv2.imshow('Frame', y_channel)

            cv2.waitKey(100)  # 更新显示

            # 保存图像数据
            npz_filename = os.path.join(output_dir, f'ir_frame_{idx:04d}.npz')
            cv2.imwrite(os.path.join(output_dir, f'ir_frame_{idx:04d}.png'), y_channel)
            np.savez(npz_filename, timestamp=timestamp, frame=y_channel)
            print(f"Saved frame {idx} with timestamp {timestamp:.2f} seconds to {npz_filename}")

        # 释放资源
        cv2.destroyAllWindows()

        return frame_timestamps, nir_start_time

    """def extract_events_and_match_frames(aedat_file, nir_start_time, frame_timestamps, output_dir, time_window_ms=30):
        # 计算时间窗口的微秒数
        time_window_us = time_window_ms * 1000
        with dv.AedatFile(aedat_file) as f:
            # 提取事件数据
            events = np.hstack([packet for packet in f['events'].numpy()])
            timestamps = events['timestamp']
            x_coords = events['x']
            y_coords = events['y']
            polarities = events['polarity']

            if 'frame' in f.names:
                for 

        file_name, _ = os.path.splitext(os.path.basename(aedat_file))
        event_start_time = file_name[-8:].replace('_', '')

        diff_seconds = time_difference(event_start_time, nir_start_time)
        print(diff_seconds)
        timestamps = (timestamps - timestamps[0]) - diff_seconds * 1000000
        print(timestamps)
        # 存储时间窗口内的事件数据
        matched_events = {}

        # 处理每一帧
        for idx, frame_time in enumerate(frame_timestamps):
            end_time = frame_time * 1000000
            start_time = end_time - time_window_us

            print(start_time, end_time)

            # 创建掩码，选择在当前时间窗口内的事件
            mask = (timestamps >= start_time) & (timestamps < end_time)
            # print(mask)

            segment_timestamps = timestamps[mask]
            segment_x_coords = x_coords[mask]
            segment_y_coords = y_coords[mask]
            segment_polarities = polarities[mask]
            print(segment_polarities.shape)
            event_volume = np.zeros(((260, 346)), dtype=np.float32)
            print(event_volume.shape)
            for xi, yi in zip(segment_x_coords, segment_y_coords):
                event_volume[yi, xi] = 1

            # 存储或处理事件数据
            matched_events[idx] = {
                'timestamps': segment_timestamps,
                'x_coords': segment_x_coords,
                'y_coords': segment_y_coords,
                'polarities': segment_polarities
            }

            # 保存为NPZ文件
            npz_filename = os.path.join(output_dir, f'events_for_frame_{idx:04d}.npz')
            cv2.imshow('s', event_volume * 255)
            cv2.waitKey(100)
            cv2.imwrite(os.path.join(output_dir, f'events_for_frame_{idx:04d}.png'), event_volume * 255)
            np.savez(npz_filename, timestamps=segment_timestamps, x_coords=segment_x_coords,
                     y_coords=segment_y_coords, polarities=segment_polarities)
            print(f"Events saved for frame {idx} to {npz_filename}")

        return matched_events"""

    def extract_events_and_match_frames(aedat_file, nir_start_time, frame_timestamps, output_dir,time_window_ms=30):
        # 计算时间窗口的微秒数
        time_window_us = time_window_ms * 1000

        # 打开AEDAT文件
        with dv.AedatFile(aedat_file) as f:
            # 提取事件数据
            events = np.hstack([packet for packet in f['events'].numpy()])
            timestamps = events['timestamp']
            x_coords = events['x']
            y_coords = events['y']
            polarities = events['polarity']


        # 文件名和时间校准
        file_name, _ = os.path.splitext(os.path.basename(aedat_file))
        event_start_time = file_name[-8:].replace('_', '')
        diff_seconds = time_difference(event_start_time, nir_start_time)
        timestamps = (timestamps - timestamps[0]) - diff_seconds * 1000000

        # 存储时间窗口内的事件数据
        matched_events = {}

        # 处理每一帧
        for idx, frame_time in enumerate(frame_timestamps):
            end_time = frame_time * 1000000
            print(frame_time)
            start_time = end_time - time_window_us

            # 创建掩码，选择在当前时间窗口内的事件
            mask = (timestamps >= start_time) & (timestamps < end_time)
            segment_timestamps = timestamps[mask]
            segment_x_coords = x_coords[mask]
            segment_y_coords = y_coords[mask]
            segment_polarities = polarities[mask]

            event_volume = np.zeros((260, 346), dtype=np.float32)
            for xi, yi in zip(segment_x_coords, segment_y_coords):
                event_volume[yi, xi] = 1

            # 保存事件数据
            npz_filename = os.path.join(output_dir, f'event1_frame_{idx:04d}.npz')
            np.savez(npz_filename, timestamps=segment_timestamps, x_coords=segment_x_coords,
                     y_coords=segment_y_coords, polarities=segment_polarities)

            # 可视化并保存事件的图像表示
            cv2.imwrite(os.path.join(output_dir, f'event1_frame_{idx:04d}.png'), event_volume * 255)

            """# 获取灰度图帧，并保存
            if idx < len(frames):
                grayscale_frame = frames[idx].image  # 假设帧包含 'image' 属性，表示灰度图
                grayscale_time = frames[idx].timestamp  # 对应的时间戳
                print(grayscale_time)

                # 保存灰度图
                grayscale_filename = os.path.join(output_dir, f'grayscale_frame_{idx:04d}.png')
                cv2.imwrite(grayscale_filename, grayscale_frame)
                print(f"Grayscale frame {idx} saved to {grayscale_filename}")"""

            print(f"Events saved for frame {idx} to {npz_filename}")


    # 递归遍历所有子目录以处理多个数据集
    for entry in os.listdir(base_dir):
        path = os.path.join(base_dir,entry)
        dir_path = os.path.join(path, 'video')
        if not dir_path:
            continue
        video_file_path = None
        aedat_file_path = None

        for f in os.listdir(dir_path):
            if f.endswith('.aedat4'):
                aedat_file_path = os.path.join(dir_path, f)
            if f.endswith('.raw'):
                video_file_path = os.path.join(dir_path, f)

        if not aedat_file_path:
            raise FileNotFoundError(f"Skipped directory {dir_path}: No .aedat4 file found.")
        if not video_file_path:
            raise FileNotFoundError(f"Skipped directory {dir_path}: No .raw file found.")

        output_ir = os.path.join(path, 'ir')
        os.makedirs(output_ir, exist_ok=True)
        output_event = os.path.join(path, 'event1')
        os.makedirs(output_event, exist_ok=True)

        # 处理视频并获取帧时间戳
        frame_timestamps, nir_start_time = process_raw_video(video_file_path, fps, output_ir)

        # 提取事件数据并与帧时间戳匹配
        extract_events_and_match_frames(aedat_file_path, nir_start_time, frame_timestamps, output_event,time_window_ms)

def save_frame_event(base_dir,time_window_us=15000):
    # 递归遍历所有子目录以处理多个数据集
    for entry in os.listdir(base_dir):
        path = os.path.join(base_dir, entry)
        dir_path = os.path.join(path, 'video')
        if not dir_path:
            continue
        for f in os.listdir(dir_path):
            if f.endswith('.aedat4'):
                aedat_file_path = os.path.join(dir_path, f)
            if f.endswith('.raw'):
                video_file_path = os.path.join(dir_path, f)
        if not aedat_file_path:
            raise FileNotFoundError(f"Skipped directory {dir_path}: No .aedat4 file found.")
        if not video_file_path:
            raise FileNotFoundError(f"Skipped directory {dir_path}: No .raw file found.")

        rgb_path = os.path.join(path, "rgb")
        if not os.path.exists(rgb_path):
            os.makedirs(rgb_path)
        event2_path = os.path.join(path, "event2")
        if not os.path.exists(event2_path):
            os.makedirs(event2_path)

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
                npz_filename = os.path.join(event2_path, f'event2_frame_{idx:04d}.npz')
                np.savez(npz_filename, timestamps=segment_timestamps, x_coords=segment_x_coords,
                         y_coords=segment_y_coords, polarities=segment_polarities)

                # 可视化并保存事件的图像表示
                event_image_filename = os.path.join(event2_path, f'event2_frame_{idx:04d}.png')
                cv2.imwrite(event_image_filename, event_volume * 255)

                # 保存灰度图帧
                grayscale_frame = frame.image  # 假设帧包含 'image' 属性，表示灰度图
                grayscale_filename = os.path.join(rgb_path, f'rgb_frame_{idx:04d}.png')
                cv2.imwrite(grayscale_filename, grayscale_frame)
                print(f"Grayscale frame {idx} saved to {grayscale_filename}")

                print(f"Events saved for frame {idx} to {npz_filename}")

def event_Completion(base_dir, img_shape=(260, 346)):

    def is_point_in_rect(point, rect_start, rect_end):
        x1, y1 = min(rect_start[0], rect_end[0]), min(rect_start[1], rect_end[1])
        x2, y2 = max(rect_start[0], rect_end[0]), max(rect_start[1], rect_end[1])

        px, py = point
        if x1 <= px <= x2 and y1 <= py <= y2:
            return True
        else:
            return False

    def read_event_npz(path, shape):
        with np.load(path, allow_pickle=True) as data:
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

    # 递归遍历所有子目录以处理多个数据集
    for entry in os.listdir(base_dir):
        path = os.path.join(base_dir, entry)
        dir_path = os.path.join(path, 'video')
        if not dir_path:
            continue

        completed_event_path = os.path.join(path, "completed_event")
        os.makedirs(completed_event_path, exist_ok=True)

        grayscale_png = sorted(glob.glob(os.path.join(os.path.join(path, "rgb"), 'rgb*.png')))
        event_files = sorted(glob.glob(os.path.join(os.path.join(path, "event"), 'event*.npz')))

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

                if warped_events is None:
                    return
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
            cv2.imwrite(os.path.join(completed_event_path, f'completed_event_frame_{i:04d}.png'), fused_event_volume * 255)

            # 保存提取到的事件数据为 npz 文件
            #y_coords, x_coords = np.nonzero(fused_event_volume)
            #np.savez(os.path.join(buquan_event_npz, f'frame_{str(i -3).zfill(4)}.npz'), x_coords=x_coords, y_coords=y_coords)

def test2(base_dir = r'./test_data', fps=50, time_window_ms=15):
    def time_difference(time1, time2):
        # 解析时分秒
        h1, m1, s1 = int(time1[:2]), int(time1[2:4]), int(time1[4:])
        h2, m2, s2 = int(time2[:2]), int(time2[2:4]), int(time2[4:])

        # 将时分秒转换为总秒数
        total_seconds1 = h1 * 3600 + m1 * 60 + s1
        total_seconds2 = h2 * 3600 + m2 * 60 + s2

        # 计算两个时间的差值
        diff_seconds = abs(total_seconds1 - total_seconds2)

        return diff_seconds

    def process_raw_video(file_path, fps, output_dir):
        # 打开原始数据文件
        with open(file_path, 'rb') as f:
            raw_data = f.read()

        file_name, _ = os.path.splitext(os.path.basename(file_path))
        nir_start_time = file_name[-9:-3]
        #print(nir_start_time)

        # 目标帧尺寸
        frame_height, frame_width, num_channels = 512, 640, 2
        frame_size = frame_height * frame_width * num_channels

        # 计算帧数
        num_frames = len(raw_data) // frame_size
        print(f"Total number of frames: {num_frames}")

        # 加载原始数据到 NumPy 数组，并确保其形状
        raw_data_array = np.frombuffer(raw_data, dtype=np.uint8)
        assert raw_data_array.size == num_frames * frame_size, "数据大小不匹配！"

        """# 创建 RGB 图像数据数组
        cvt_data = np.zeros([num_frames, frame_height, frame_width, 3], dtype=np.uint8)"""

        # 计算每帧的时间间隔
        time_interval = 1 / fps  # 每帧时间间隔（秒）

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 存储帧的时间戳
        frame_timestamps = []

        # 处理每一帧
        first = 0
        for idx in range(0,num_frames,5):
            print(idx)
            """# 转换 YUV422 到 RGB
            cvt_data[idx, ...] = cv2.cvtColor(final_data[idx, ...], cv2.COLOR_YUV2RGB_Y422)"""

            # 提取当前帧的 YUV422 数据
            start_idx = idx * frame_size
            end_idx = start_idx + frame_size
            frame_data = raw_data_array[start_idx:end_idx]

            # 重塑为 (height, width, 2) 以匹配 YUV422 格式
            yuv_frame = frame_data.reshape((frame_height, frame_width, 2))

            # 只提取 Y 通道，Y 通道是每个像素的亮度值
            y_channel = yuv_frame[:, :, 0]

            # 计算时间戳
            timestamp = idx * time_interval
            frame_timestamps.append(timestamp)

            """# 显示图像
            gray_data = cv2.cvtColor(cvt_data[idx, ...][:256, :320], cv2.COLOR_RGB2GRAY)"""

            cv2.imshow('Frame', y_channel)

            cv2.waitKey(100)  # 更新显示

            # 保存图像数据
            #npz_filename = os.path.join(output_dir, f'ir_frame_{idx/2:04d}.npz')
            cv2.imwrite(os.path.join(output_dir, f'ir_frame_{int(first):04d}.png'), y_channel)
            first +=1
            #np.savez(npz_filename, timestamp=timestamp, frame=y_channel)
            #print(f"Saved frame {idx} with timestamp {timestamp:.2f} seconds to {npz_filename}")

        # 释放资源
        cv2.destroyAllWindows()
        print(frame_timestamps)

        return frame_timestamps, nir_start_time


    def extract_events_and_match_frames(aedat_file, nir_start_time, ir_timestamps, output_dir,output_rgb,
                                        time_window_ms=30):
        # 计算时间窗口的微秒数
        time_window_us = time_window_ms * 1000

        # 打开AEDAT文件
        with dv.AedatFile(aedat_file) as f:
            # 提取事件数据
            events = np.hstack([packet for packet in f['events'].numpy()])
            timestamps = events['timestamp']
            x_coords = events['x']
            y_coords = events['y']
            polarities = events['polarity']
            frames = [frame for frame in f['frames']]
            frames_timestamps1 = [frame.timestamp for frame in frames]
            #for n in range(len(frames_timestamps)):
            #    print(frames_timestamps[n]-frames_timestamps[0])
            #print(len(frames_timestamps))


        frames_timestamps = np.array(frames_timestamps1)
        # 文件名和时间校准
        file_name, _ = os.path.splitext(os.path.basename(aedat_file))
        event_start_time = file_name[-8:].replace('_', '')
        diff_seconds = time_difference(event_start_time, nir_start_time)
        print(diff_seconds,event_start_time,nir_start_time)
        timestamps = (timestamps - timestamps[0]) - diff_seconds * 1000000
        #print(frames_timestamps)
        frames_timestamps = (frames_timestamps - frames_timestamps[0]) - diff_seconds * 1000000
        print(frames_timestamps)

        #print(len(frames_timestamps))
        for x in range(len(frames_timestamps)):
            if frames_timestamps[x]>=0:
                first_frame_id = x
                break
        # 存储时间窗口内的事件数据
        matched_events = {}
        print(first_frame_id,'-----------')

        # 处理每一帧
        first = 0
        for idx, frame_time in enumerate(ir_timestamps):
            grayscale_frame = frames[first_frame_id].image  # 假设帧包含 'image' 属性，表示灰度图
            first_frame_id += 1
            # grayscale_time = frames[idx].timestamp  # 对应的时间戳
            # print(grayscale_time)

            # 保存灰度图
            grayscale_filename = os.path.join(output_rgb, f'rgb_frame_{first:04d}.png')
            first +=1
            cv2.imwrite(grayscale_filename, grayscale_frame)
            print(f"Grayscale frame {idx} saved to {grayscale_filename}")

            end_time = frame_time * 1000000
            #print(frame_time)
            start_time = end_time - time_window_us

            # 创建掩码，选择在当前时间窗口内的事件
            mask = (timestamps >= start_time) & (timestamps < end_time)
            segment_timestamps = timestamps[mask]
            segment_x_coords = x_coords[mask]
            segment_y_coords = y_coords[mask]
            segment_polarities = polarities[mask]

            event_volume = np.zeros((260, 346), dtype=np.float32)
            for xi, yi in zip(segment_x_coords, segment_y_coords):
                event_volume[yi, xi] = 1

            # 保存事件数据
            npz_filename = os.path.join(output_dir, f'event_frame_{idx:04d}.npz')
            np.savez(npz_filename, timestamps=segment_timestamps, x_coords=segment_x_coords,
                     y_coords=segment_y_coords, polarities=segment_polarities)

            # 可视化并保存事件的图像表示
            cv2.imwrite(os.path.join(output_dir, f'event_frame_{idx:04d}.png'), event_volume * 255)
        print(first_frame_id-1)
        print(frames_timestamps[first_frame_id-1])




    # 递归遍历所有子目录以处理多个数据集
    for entry in os.listdir(base_dir):
        path = os.path.join(base_dir, entry)
        dir_path = os.path.join(path, 'video')
        if not dir_path:
            continue
        video_file_path = None
        aedat_file_path = None

        for f in os.listdir(dir_path):
            if f.endswith('.aedat4'):
                aedat_file_path = os.path.join(dir_path, f)
            if f.endswith('.raw'):
                video_file_path = os.path.join(dir_path, f)

        if not aedat_file_path:
            raise FileNotFoundError(f"Skipped directory {dir_path}: No .aedat4 file found.")
        if not video_file_path:
            raise FileNotFoundError(f"Skipped directory {dir_path}: No .raw file found.")

        output_ir = os.path.join(path, 'ir')
        os.makedirs(output_ir, exist_ok=True)
        output_event = os.path.join(path, 'event')
        os.makedirs(output_event, exist_ok=True)
        output_rgb = os.path.join(path, 'rgb')
        os.makedirs(output_rgb, exist_ok=True)

        # 处理视频并获取帧时间戳
        frame_timestamps, nir_start_time = process_raw_video(video_file_path, fps, output_ir)

        print(nir_start_time)

        # 提取事件数据并与帧时间戳匹配
        extract_events_and_match_frames(aedat_file_path, nir_start_time, frame_timestamps, output_event,output_rgb,
                                        time_window_ms)

def main1():
    base_dir = r'./test_data'
    event_nir_fusion(base_dir)
    save_frame_event(base_dir)

def main2():
    base_dir = r'./test_data'
    event_Completion(base_dir)

def main3():
    base_dir = r'C:/Users/zsq/Desktop/d'
    test2(base_dir)
    event_Completion(base_dir)





if __name__ == '__main__':
    main3()