function [train_data,test_data,test_label] = divide_data(n)  
    train_data = zeros(40,10304);   % 将图像的像素展平是10304维的向量
    test_data = [];
    test_label=[];
    
    % 遍历每个人物的每张照片
    for i = 1:40
        random_num = randperm(10, n);  % 从10张人脸照片中选出n张作为训练集
        for j = 1:10
            img = reshape(double(imread(['./ORL_faces/s',num2str(i),'/',num2str(j),'.pgm'])),1,10304); 
            if ismember(j, random_num)
                train_data(i,:) = train_data(i,:) + img;
            else
                test_data = [test_data; img];
                test_label = [test_label;i];   % 记录训练集的标签
            end
        end
    end
    train_data = train_data / n;       % 计算几张图片的平均值
end

