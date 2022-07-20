img = imread('barb.png'); % 读取图片
[height, width] = size(img); % 获取图片的长和宽

% subplot(2, 2, 1);
% imshow(img);
figure();

f = double(img); % 读取图像的灰度值
f = Centralize(f); % 对图像进行中心化处理
F = fft2(f); % 傅立叶变换

% 输出频域上的图像
% margin= log(abs(F));
% imshow(margin,[])
% pause

H = Butterworth(10,1, height, width); % 获取butterworth的滤波器
g = real(ifft2(H .* F));% 做一个点乘操作然后反傅立叶变换
g = Centralize(g);% 再做个中心化将图像复原成原来的样子
subplot(2,2,1);
imshow(uint8(g));
title('D0 = 10  n=1');

H = Butterworth(20,1, height, width);
g = real(ifft2(H .* F));
g = Centralize(g);
subplot(2,2,2);
imshow(uint8(g));
title('D0 = 20  n=1');

H = Butterworth(40,1, height, width);
g = real(ifft2(H .* F));
g = Centralize(g);
subplot(2,2,3);
imshow(uint8(g));
title('D0 = 40  n=1');

H = Butterworth(80,1, height, width);
g = real(ifft2(H .* F));
g = Centralize(g);
subplot(2,2,4);
imshow(uint8(g));
title('D0 = 80  n=1');