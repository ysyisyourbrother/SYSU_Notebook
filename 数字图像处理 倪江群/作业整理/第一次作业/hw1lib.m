%函数histeq（）进行直方图均衡化处理
I=imread('river.jpg');
J=histeq(I,256);  %直方图均衡化
% figure,
% subplot(121),imshow(uint8(I));
% title('原图')
% subplot(122),imshow(uint8(J));
% title('均衡化后')
figure,
subplot(121),imhist(I,256);
% subplot(122),imhist(I,256);
title('原图像直方图');
subplot(122),imhist(J,256);
title('均衡化后的直方图');