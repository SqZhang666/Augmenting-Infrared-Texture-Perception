clc; clear; close all;
warning('off')

addpath ir_event   % type of multi-modal data

% 读取图像
str1 = 'labeled_bw1.png';   % 图像 1
str2 = 'frame_0000.png';   % 图像 2
str3 = 'filtered_events_for_frame_0004.png';   % 图像 2
im1 = im2uint8(imread(str1));
im1(im1 ~= 0) = 100;
im2 = im2uint8(imread(str2));
im3 = im2uint8(imread(str3));

% 确保两张图像都是 RGB 格式
if size(im1, 3) == 1
    temp = im1;
    im1(:, :, 1) = temp;
    im1(:, :, 2) = temp;
    im1(:, :, 3) = temp;
end

if size(im2, 3) == 1
    temp = im2;
    im2(:, :, 1) = temp;
    im2(:, :, 2) = temp;
    im2(:, :, 3) = temp;
end

% 选取图二的局部区域进行放大 (ROI 选择)
roi2 = [180, 200, 100, 100];  % im2 中的感兴趣区域

% 提取局部区域并进行放大 (zoom in)
zoom_factor = 2;  % 放大倍数
%im1_cropped = imcrop(im1, roi1);
im2_cropped = imcrop(im2, roi2);
im2_zoomed = imresize(im2_cropped, zoom_factor);

% RIFT 特征点检测与描述 (图一和放大后的图二局部)
[des_m1, des_m2] = RIFT_no_rotation_invariance(im1, im2_zoomed, 10, 2, 30);

% 最近邻特征点匹配
[indexPairs, matchmetric] = matchFeatures(des_m1.des, des_m2.des, 'MaxRatio', 1, 'MatchThreshold', 10);
matchedPoints1 = des_m1.kps(indexPairs(:, 1), :);
matchedPoints2 = des_m2.kps(indexPairs(:, 2), :);

% 将缩放后的特征点坐标映射回全局坐标：
matchedPoints2_global = (matchedPoints2 - 1) / zoom_factor + roi2(1:2); % 从局部ROI返回全局坐标

% 去除重复点
[matchedPoints2_global, IA] = unique(matchedPoints2_global, 'rows');
matchedPoints1 = matchedPoints1(IA, :);

% 剔除离群点
disp('outlier removal');
H = FSC(matchedPoints1, matchedPoints2_global, 'affine', 2);
Y_ = H * [matchedPoints1'; ones(1, size(matchedPoints1, 1))];
Y_(1, :) = Y_(1, :) ./ Y_(3, :);
Y_(2, :) = Y_(2, :) ./ Y_(3, :);
E = sqrt(sum((Y_(1:2, :) - matchedPoints2_global').^2));
inliersIndex = E < 3;
cleanedPoints1 = matchedPoints1(inliersIndex, :);
cleanedPoints2 = matchedPoints2_global(inliersIndex, :);

disp('registration result')
figure; showMatchedFeatures(im1, im2, cleanedPoints1, cleanedPoints2, 'montage');

image_fusion(im2, im3, double(H));