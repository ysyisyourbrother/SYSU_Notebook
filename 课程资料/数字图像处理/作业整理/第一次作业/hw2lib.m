clear
I=imread('EightAM.png');%读取图像
Imatch=imread('LENA.png');%读取匹配图像
Jmatch=imhist(Imatch);%获取匹配图像直方图
Iout=histeq(I,Jmatch);%直方图匹配
figure;%显示原图像、匹配图像和匹配后的图像
subplot(1,3,1),imshow(I);title('原图像');
subplot(1,3,2),imshow(Imatch);title('匹配图像');
subplot(1,3,3),imshow(Iout);title('匹配之后图像');
figure;%显示原图像、匹配图像和匹配后图像的直方图
subplot(3,1,1),imhist(I,256);title('原图像直方图');
subplot(3,1,2),imhist(Imatch,256);title('匹配图像图像直方图');
subplot(3,1,3),imhist(Iout,256);title('匹配之后图像直方图');
       