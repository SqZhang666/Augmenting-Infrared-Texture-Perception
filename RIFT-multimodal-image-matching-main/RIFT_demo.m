clc; clear; close all;
warning('off')

addpath ir_event   % type of multi-modal data

% 设置文件夹路径
event_png_path = 'D:\Learning\KEY\CV\code\demo\10_19\13\deal_event_img1';%要补的事件图片
ir_path = 'D:\Learning\KEY\CV\code\demo\10_19\13\ir_png';%红外图片
bw_event_path = 'D:\Learning\KEY\CV\code\demo\10_19\13\deal_grayscale';%要补的事件数据对应的RGB图片
mask_path = 'D:\Learning\KEY\CV\code\demo\10_19\13\a1'; %仅包含纹理部分的二值图
event_files = dir(fullfile(event_png_path, '*.png'));
ir_files = dir(fullfile(ir_path, '*.png'));
bw_event_files = dir(fullfile(bw_event_path,'*.png'));
mask_files = dir(fullfile(mask_path,'*.png'));
num_images = length(event_files);

% 确保文件数量一致
if length(event_files) ~= length(ir_files) || length(event_files) ~= length(bw_event_files)
    error('文件夹中的文件数量不一致，请检查文件夹内容');
end

% 读取图像并处理
for i = 1:num_images-1
    
    %str = ['./fusion_image/', num2str(i), '.png'];
    %bw_event = a(event,str);
    
    bw_event = im2uint8(imread(fullfile(bw_event_path, bw_event_files(i).name)));
    %bw_event(bw_event ~= 0) = 100;  % 将非零值设置为100 这个是最开始二值化事件图片的时候用的
    
    
    ir = im2uint8(imread(fullfile(ir_path, ir_files(i).name)));
  

    % 确保两张图像都是 RGB 格式
    if size(bw_event, 3) == 1
        temp = bw_event;
        bw_event(:, :, 1) = temp;
        bw_event(:, :, 2) = temp;
        bw_event(:, :, 3) = temp;
    end

    if size(ir, 3) == 1
        temp = ir;
        ir(:, :, 1) = temp;
        ir(:, :, 2) = temp;
        ir(:, :, 3) = temp;
    end
    
    roi1 = [2, 36, 150, 170];
    roi2 = [165, 200, 100, 120];  % im2 中的感兴趣区域
    
    % 提取局部区域并进行放大 (zoom in)
    zoom_factor = 2;  % 放大倍数
    im1_cropped = imcrop(bw_event, roi1);
    im1_zoomed = imresize(im1_cropped, 1);
    im2_cropped = imcrop(ir, roi2);
    im2_zoomed = imresize(im2_cropped, zoom_factor);
    %figure;
    %imshow(im2_zoomed);
    

    % RIFT 特征点检测与描述 (图一和放大后的图二局部)
    [des_m1, des_m2] = RIFT_no_rotation_invariance(im1_zoomed, im2_zoomed, 10, 2, 30);

    % 最近邻特征点匹配
    [indexPairs, matchmetric] = matchFeatures(des_m1.des, des_m2.des, 'MaxRatio', 1, 'MatchThreshold', 10);
    matchedPoints1 = des_m1.kps(indexPairs(:, 1), :);
    matchedPoints2 = des_m2.kps(indexPairs(:, 2), :);

    % 将缩放后的特征点坐标映射回全局坐标：
    matchedPoints1_global = (matchedPoints1 - 1) + roi1(1:2); % 从局部ROI返回全局坐标
    matchedPoints2_global = (matchedPoints2 - 1) / zoom_factor + roi2(1:2); % 从局部ROI返回全局坐标

    % 去除重复点
    [matchedPoints2_global, IA] = unique(matchedPoints2_global, 'rows');
    matchedPoints1_global = matchedPoints1_global(IA, :);

    % 剔除离群点
    disp('outlier removal');
    H = FSC(matchedPoints1_global, matchedPoints2_global, 'affine', 2);
    Y_ = H * [matchedPoints1_global'; ones(1, size(matchedPoints1_global, 1))];
    Y_(1, :) = Y_(1, :) ./ Y_(3, :);
    Y_(2, :) = Y_(2, :) ./ Y_(3, :);
    E = sqrt(sum((Y_(1:2, :) - matchedPoints2_global').^2));
    inliersIndex = E < 3;
    cleanedPoints1 = matchedPoints1_global(inliersIndex, :);
    cleanedPoints2 = matchedPoints2_global(inliersIndex, :);

    disp('registration result');
    %figure; showMatchedFeatures(bw_event, ir, cleanedPoints1, cleanedPoints2, 'montage');
    %str=['.\save_image3\','matched_image',num2str(i),'.png'];
    %saveas(gcf, str);

    event = im2uint8(imread(fullfile(event_png_path, event_files(i).name)));
    %figure;
    %imshow(event);
    disp(['Event file: ', event_files(i).name, ' | IR file: ', ir_files(i).name, ' | BW Event file: ', bw_event_files(i).name]);
    image_fusion(ir, event, double(H),i,1);  % 进行图像融合
    
    mask = im2uint8(imread(fullfile(mask_path, mask_files(i).name)));
    mask = 255-mask;
    %figure;
    %imshow(event);
    image_fusion(ir, mask, double(H),i,2);  % 进行图像融合
    
    close all

end
