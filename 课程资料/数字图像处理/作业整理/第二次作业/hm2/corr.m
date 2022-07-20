% 转变成double类型的
image = double(imread('car.png'));
filter = double(imread('wheel.png'));

%获取图像的长度和宽度
M = size(image,1);
N = size(image,2);

%获取匹配模版的长度和宽度
m=size(filter,1);
n=size(filter,2);
row_middle=(m-1)/2+1;
col_middle=(n-1)/2+1;

%定义一个全0的矩阵和原来的图像大小一致
%为了让卷积操作结束后图像大小一致，因此需要先拓展图片。
Corr_image = zeros([M,N]);
pad_image = padarray(image, [(m-1)/2, (n-1)/2]);

%卷积操作
for i = (1+(m-1)/2):(M+(m-1)/2)
    for j = (1+(n-1)/2):(N+(n-1)/2)
        % 计算一个filter中的卷积和  此部分也可以换成矩阵的点乘形式
        corr_sum=0;normal_sum=0;
        for x = -(m-1)/2:(m-1)/2
           for y = -(n-1)/2:(n-1)/2
               corr_sum=corr_sum+pad_image(i+x,j+y)*filter(row_middle+x,col_middle+y); % 计算公式中的分子部分 卷积和
               normal_sum=normal_sum+pad_image(i+x,j+y); % 计算公式中的分母部分 归一化的和
           end
        end
        Corr_image(i-(m-1)/2,j-(n-1)/2)=corr_sum/normal_sum;
    end
end

% 归一化矩阵 并找出阈值大于一定的点
Corr_image=Corr_image./max(max(Corr_image));
res=[];
for i =1+(m-1)/2:M-(m-1)/2
    for j = 1+(n-1)/2:N-(n-1)/2
        if Corr_image(i,j) >0.86
            % 对指向相同轮子的点进行去重
            flag=1;a=size(res,1);
            for index=1:a
               if sqrt((res(index,1)-i)^2+(res(index,2)-j)^2)<10
                   flag=0;break;
               end
            end
            if flag==1
                res=[res;i,j];
            end
            wheel=image(i-(m-1):i+(m-1),j-(n-1):j+(n-1)); % 截取出轮子部分的图像
            imshow(wheel./max(max(wheel)));
            
        end
    end
end
res


% 输出矩阵到文件中
% fid=fopen('corr_mat.txt','w');
% for i=1:M
%     for j=1:N
%         if j==N
%             fprintf(fid,'%d\r\n',Corr_image(i,j));%如果是最后一个，就换行
%         else
%             fprintf(fid,'%d\r\t',Corr_image(i,j));%如果不是最后一个，就tab
%         end
%     end
% end
% fclose(fid);



