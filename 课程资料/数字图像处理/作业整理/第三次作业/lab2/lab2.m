img = imread('office.jpg');
img = double(rgb2gray(img));
[height, width] = size(img);
figure();

% 绘制原图像
subplot(2, 2, 1);
imshow(img,[]);% 自动调整住距的方位到0-1之间
title('原图像');

% 绘制同态滤波后图像
% gammaH = 2;
% gammaL = 0.25;
% C = 1;
% D0 = 1;
% num = 2;
% while 1
%     F = fft2(log(img + 1));% 防止有值为0
%     H = HomomorphicFiltering(gammaH, gammaL, C, D0, height, width);% 对频域图像使用高通高斯滤波器
%     g = real(exp(ifft2(H .* F)));% 反傅立叶变换的结果由于四舍五入还是复数
%     new_img = maxmin(g);
%     subplot(2,3,num); imshow(new_img);title(['同态滤波(D0 = ',num2str(D0),')']);
%     D0=D0*10;
%     num=num+1;
%     if D0==100000
%         break
%     end
% end

% 绘制同态滤波后图像
gammaH = 2;
gammaL = 0.25;
C = 1;
D0 = 1000;
F = fft2(log(img + 1));% 防止有值为0
H = HomomorphicFiltering(gammaH, gammaL, C, D0, height, width);% 对频域图像使用高通高斯滤波器
g = real(exp(ifft2(H .* F)));% 反傅立叶变换的结果由于四舍五入还是复数
new_img = maxmin(g);
subplot(2,2,2); imshow(new_img);title('同态滤波(D0 = 1000)');


% 对比高通滤波器butterworth效果
F = fft2(Centralize(img));
D0 = 1;
H = Butterworth(D0, height, width);
g = real(ifft2(H .* F));
g = Centralize(g);
subplot(2,2,3); imshow(uint8(g));
title('高通滤波(D0 = 1)');
