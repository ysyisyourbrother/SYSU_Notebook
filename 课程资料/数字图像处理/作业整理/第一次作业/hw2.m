% 读取图片并得到灰度图
Origin      = imread('EightAM.png');
Refer   = imread('LENA.png');
histO    = imhist(Origin);          
histRefer = imhist(Refer);

% 使用cumsum函数计算累积和
cum_sum_O     = cumsum(histO) / numel(Origin);
cum_sum_R  = cumsum(histRefer) / numel(Refer);

% 计算映射关系
map  = zeros(1,256);
for idx = 1 : 256
    % 找到和原图均衡化后的每个灰度级
    a=abs(cum_sum_O(idx) - cum_sum_R); % 找和cdf(idx)最接近的灰度值的下标。
    [b,index] = min(a);
    map(idx)    = index-1;
end
 
% 下标是从1到256 灰度级是0到255，因此要加一，同时提取多列
OMatch = map(uint16(Origin)+1);

%显示图像
figure;
subplot(1,3,1),imshow(Origin,[]);title('原图像');
subplot(1,3,2),imshow(Refer,[]);title('匹配图像');
subplot(1,3,3),imshow(OMatch,[]);title('匹配之后图像');

%显示直方图
figure;
subplot(3,1,1),imhist(Origin,256);title('原图像直方图');
subplot(3,1,2),imhist(Refer,256);title('匹配图像直方图');
subplot(3,1,3),imhist(uint8(OMatch),256);title('匹配之后图像直方图');