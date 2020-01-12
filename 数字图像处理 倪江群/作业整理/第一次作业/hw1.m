clc;
clear;
fx = imread('river.jpg');

subplot(2,2,1);imshow(fx);title('灰度图');
subplot(2,2,3);imhist(fx);title('灰度图');
 
[R, C] = size(fx);

% 统计每个像素值出现次数
% 因为下标是1到256而直方图是0到255
cum = zeros(1, 256);
for i = 1 : R
    for j = 1 : C
        cum(fx(i, j) + 1) = cum(fx(i, j) + 1) + 1;
    end
end


cum = double(cum);
% 求累计概率，得到累计直方图
for i = 2 : 256
    cum(i) = (cum(i - 1) + cum(i));
end
 
for i = 1 : 256
    cum(i) = cum(i)/(R*C) * 255;
end

% 映射
fy = double(fx);
for i = 1 : R
    for j = 1 : C
        fy(i, j) = cum(fy(i, j) + 1);
    end
end
 
% 输出仍然要记得改数据类型
fy = uint8(fy);
subplot(2,2,2);imshow(fy,[]);title('直方图均衡化');
subplot(2,2,4);imhist(fy);title('直方图均衡化');