clc;
clear all;
close all;
image = imread('sport car.pgm');
%计算图像的长度和宽度
M = size(image, 1); N = size(image, 2);
% 定义滤波器的长度和宽度
m = 3;n = 3;

% 产生两个均匀分布
t1 = rand([M,N]) * 255;
t2 = rand([M,N]) * 255;

% 生成椒盐噪声的图像
noise_image = image;
for i = 1:M
    for j = 1:N
        if image(i,j) > t1(i,j)
            noise_image(i,j) = 255;
        elseif image(i,j) < t2(i,j)
            noise_image(i,j) = 0;
        else
            noise_image(i,j)=image(i,j);
        end
        
    end
end

% 进行中值滤波操作
pad_image = padarray(noise_image, [(m-1)/2, (n-1)/2]);% 先对原来的图像进行padding
median_image = uint8(zeros([M,N]));
for i = 1:M
    for j = 1:N
        tmp=pad_image(i:i+m-1, j:j+n-1);
        tmp = sort(tmp(:));
        median_image(i,j) = tmp(5);
    end
end


% 用toolbox的medfilt2函数
median_lib = medfilt2(noise_image, [3,3]);

% 绘制图像
figure
subplot(2,2,1), imshow(image);
title("原图像");
subplot(2,2,2), imshow(noise_image);
title("椒盐噪声图像");
subplot(2,2,3), imshow(median_image);
title("中值滤波图像");
subplot(2,2,4), imshow(median_lib);
title("调库后结果图像");