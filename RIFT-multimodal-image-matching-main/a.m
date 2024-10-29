function result=a(im, output_image_path)
    % process_image 处理图像，去除噪声并标记连通区域
    %
    % 输入:
    %   input_image_path  输入图像的文件路径
    %   output_image_path 输出处理后图像的保存路径

    % 创建一个与输入图像相同大小的逻辑数组
    result = true(size(im));
    
    % 将灰度图像二值化
    bw = imbinarize(im);
    
    % 创建一个圆盘形结构元素
    se = strel('disk', 1);
    
    % 对二值图像进行开运算去除噪声
    bw_cleaned = imopen(bw, se);
    
    % 显示原始图像和清理后的二值图像
    %figure;
    %subplot(1, 2, 1);
    %imshow(im);
    %title('Original Image');
    
    %subplot(1, 2, 2);
    %imshow(bw_cleaned);
    %title('Cleaned Binary Image');
    
    % 给每个连通区域进行标记
    labeled_bw = bwlabel(bw_cleaned);
    
    % 将连通区域对应的像素设为 0 
    result(labeled_bw == 1) = 0; 
    
    % 显示处理后的图像
    %figure;
    %imshow(result); 
    title('Labeled Connected Components');
    
    % 保存处理后的图像
    imwrite(result,output_image_path , 'png');
end


