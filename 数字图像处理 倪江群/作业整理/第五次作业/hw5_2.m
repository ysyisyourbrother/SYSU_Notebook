clc
clear

A = im2double(imread('blobz1.png'));
figure;
subplot(2,2,1);imshow(A);title("原图1")
subplot(2,2,2);imshow(segmentation(A));title("图1直接处理后")

B = im2double(imread('blobz2.png'));
subplot(2,2,3);imshow(B);title("图2")
subplot(2,2,4);imshow(segmentation(B));title("图2直接处理后")
[m, n] = size(B);

figure
segments = [2,4,6,8];
[sm,sn]=size(segments);
R=B; % 定义一个一样大的数组用来装生成的目标
for s=1:sn
    segment=segments(s);
    for i = 1:segment
        for j = 1:segment
            PB = segmentation(B(floor((i-1)/segment*m+1):min(floor(i/segment*m),m), floor((j-1)/segment*n+1):min(floor(j/segment*n),n)));
            R(floor((i-1)/segment*m)+1:min(floor(i/segment*m),m), floor((j-1)/segment*n)+1:min(floor(j/segment*n),n)) = PB;
        end
    end
    subplot(2,2,s);imshow(R);title("图2分块处理")
end

