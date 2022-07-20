function accuracy = Identify(train_data, test_data,test_label)
    [coeff, ~, ~] = pca(train_data); % 使用pca训练数据
    train_pca = train_data * coeff;  % 把训练的数据映射到k维空间上
    % 进行图像识别
    [M, N] = size(train_data); % 读取数据库中数据集大小
    [m, n] = size(test_data); % 读取等待测试的图片数据集大小
    count = 0;  % 统计正确预测的图片数
    for i = 1:m
        similarity = [];
        test_pca = test_data(i,:) * coeff;
        for k = 1:M   % 遍历所有人物图像的特征向量 计算和待检测图片的距离
            similarity = [similarity,norm(train_pca(k,:) - test_pca, 2)];
        end
        [~, index] = min(similarity);   % 从所有可能种类中找出相似度最高的向量
        if index == test_label(i)
            count = count+1;
        end
    end
    accuracy = count / m;
end