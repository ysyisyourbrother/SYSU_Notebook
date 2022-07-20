image = imread('book_cover.jpg');
[m,n] = size(image);
image = double(image);  

F1 = fftshift(fft2(image));  % 傅里叶变换并中心化

%% 第一二问
% 构建模糊滤波器
a = 0.1;
b = 0.1;
T = 1;
for u = 1:m
    for v = 1:n
        x = pi * ((u-m/2)*a + (v-n/2)*b); % 因为做了中心化，原点被平移到了m/2 n/2
        if (x == 0)
            H(u,v) = T;
        else
            H(u,v) = (T / x) * sin(x) * exp(-1i*x);
        end
    end
end
% 对原图进行模糊滤波器的滤波 并傅里叶反变换
res1 = real(ifft2(ifftshift(F1 .* H))); %去中心化后再做一个反傅里叶变换后取实部

% 使用minmax将图像变成0到255之间
minmax = max(res1(:)) - min(res1(:));
res1 = uint8((res1 - min(res1(:))).*255./minmax);

stand_normal_noise = randn(m,n); % randn用来生成一个随机的标准正态分布矩阵
normal_noise = stand_normal_noise*sqrt(500)+0;% 对标准正态分布进行一个线性变换得到目标正态分布
res2 = res1 + uint8(normal_noise);



%% 第三问
% 对上面的结果进行傅里叶变换
F3_1 = fftshift(fft2(res1));
F3_2 = fftshift(fft2(res2));

% 逆滤波 在反傅里叶变换 后取实部
% 这里要加一防止除以0的问题出现
res3_1 = real(ifft2(ifftshift(F3_1 ./ (H+1)))); 
minmax = max(res3_1(:)) - min(res3_1(:));
res3_1 = uint8((res3_1 - min(res3_1(:))) .* 255 ./ minmax);

res3_2 = real(ifft2(ifftshift(F3_2 ./ (H+1))));
minmax = max(res3_2(:)) - min(res3_2(:));
res3_2 = uint8((res3_2 - min(res3_2(:))) .* 255 ./ minmax);

K_list=[0.005,0.01,0.05,0.1];
[km,kn]=size(K_list);

figure
subplot(2,3,1), imshow(uint8(image)), title("原图像");
subplot(2,3,2), imshow(res1), title("运动模糊图像");
subplot(2,3,3), imshow(res2), title("运动模糊加噪图像");
subplot(2,3,4), imshow(res3_1), title("逆滤波处理运动模糊图像");
subplot(2,3,5), imshow(res3_2), title("逆滤波处理运动模糊加噪图像");

%% 第四问
figure
for i =1:kn
    K = K_list(i);
    Wiener = (1./H).*(abs(H).^2) ./ (abs(H).^2 + K);
    F4 = fftshift(fft2(res2)); % 将blurry图像傅里叶变换
    res4 = real(ifft2(ifftshift(F4 .* Wiener)));
    range = max(res4(:)) - min(res4(:));
    res4_1 = uint8((res4 - min(res4(:))) .* 255 ./ range);
    res4(:,:,i)=res4_1;
    subplot(2,2,i), imshow(uint8(res4(:,:,i))), title(["维纳滤波当K=",num2str(K)]);
end

