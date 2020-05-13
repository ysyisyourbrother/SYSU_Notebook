clc;
clear;

accuracy = []
n = 5;  % 每个人的10张照片中用几个作为训练集
% 因为每次都是随机选择图像的，所以可以多尝试几次选出正确率最高的情况
times = 100; 
for i = 1:times
    [train_data,test_data,test_label] = divide_data(n);
    accuracy = [accuracy, Identify(train_data, test_data,test_label)];
end
max_acc = max(accuracy)  % 找出最大的正确率
meanAccu = mean(accuracy);
x = 1:times;
plot(x, accuracy,'-o');
xlabel('times');
ylabel('accuracy');