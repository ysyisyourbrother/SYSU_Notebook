import os
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import cnn
import resnet18

## hyper parameters
EPOCHS = 100
BATCH_SIZE = 50
LR = 0.001

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = cnn.CNN().to(device)
# net = torchvision.models.resnet18(num_classes=10).to(device)
# net = resnet18.ResNet18(10).to(device)

# 对图像进行变换，一般需要把PIL图像转成tensor，且归一化
transform_train = transforms.Compose([
    # transforms.Resize((224,224)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    # transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


# 定义训练数据集
## 该数据集是彩色图像，channel为3
train_data = torchvision.datasets.CIFAR10(
    root='data\cifar10\cifar-10-python',
    train=True,
    transform=transform_train
)
# 使用Data的loader方便在训练时使用mini_batch
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


# 定义测试数据集
test_data = torchvision.datasets.CIFAR10(
    root='data\cifar10\cifar-10-python', 
    train=False,
    transform=transform_test
)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)



def showImg(data_loader, count):
    '''
    显示loader中的图像
    '''
    cnt = 1
    for image, label in data_loader: # 每次返回一个batch的样本
        if cnt > count:
            break
        print(label) #label是一个BATCH_SIZE大小的向量，表示各个样本的类别
        img = image[0].numpy().transpose(1,2,0) # 把channel维度放到最后
        plt.imshow(img)
        plt.show()
        cnt += 1
        


def train():
    # 定义优化器和损失函数
    # optimizer = torch.optim.Adam(params=net.parameters(), lr=LR)
    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    # optimizer = torch.optim.SGD(net.parameters(), lr=LR)
    loss_function = nn.CrossEntropyLoss() # 交叉熵损失函数常用于分类

    best_accuracy = .0 # 记录最优正确率的模型参数，保存该组参数

    with open('cnn_batch50_BN_SGD_log_file.txt', 'w') as log_file:
        # 下面对网络训练
        for epoch in range(EPOCHS):
            net.train()
            accu_loss = .0
            for step, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                optimizer.zero_grad()                   # 每次优化前都要清空梯度
                output = net(batch_x)                   # 网络的输出
                loss = loss_function(output, batch_y)   # 计算损失值
                loss.backward()                         # 反向传播误差，产生梯度
                optimizer.step()                        # 更新参数

                # 训练完一个batch后，记录累积损失值
                accu_loss += loss.item() # item方法得到tensor中的元素值
                if (step+1) % 200 == 0:   #每2000次batch_size的输入，打印一次结果
                    print('[%d, %5d] loss: %.3f' %(epoch + 1, step + 1, accu_loss / 2000)) #平均loss
                    # log信息
                    log_file.write('[%d, %5d] loss: %.3f\n' %(epoch + 1, step + 1, accu_loss / 2000))
                    accu_loss = .0


            # 一个epochc训练完后，用测试集检查正确率
            net.eval()
            with torch.no_grad():   # 测试集不能影响模型参数，禁止梯度反向传播
                correct = 0 # 记录整个测试集中预测正确的样本数量
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                    output = net(batch_x)
                    _, pred_label = torch.max(output, 1)
                    correct += pred_label.eq(batch_y).sum().item()
                accuracy = correct / len(test_data) * 100.0
                print('EPOCH:%d: %.3f%%' %(epoch, accuracy))
                # log信息
                log_file.write('EPOCH:%d: %.3f%%\n' %(epoch, accuracy))
                
                # 若准确率提高了，则记录模型参数
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(net.state_dict(), 'cnn_batch50_BN_SGD_model.pth')
        log_file.write('Best Accuracy: %.3f\n' %best_accuracy)
        

def test():
    '''
    不需要训练模型，直接加载预训练好的参数来初始化模型
    注意要调用eval()
    '''
    net.load_state_dict(torch.load('model.pth'))
    net.eval()
    with torch.no_grad():   # 测试集不能影响模型参数，禁止梯度反向传播
        correct = 0 # 记录整个测试集中预测正确的样本数量
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            output = net(batch_x)
            _, pred_label = torch.max(output, 1)
            correct += pred_label.eq(batch_y).sum().item()
        accuracy = correct / len(test_data) * 100.0
        print('EPOCH:%d: %.3f%%' %(epoch, accuracy))



if __name__ == "__main__":
    train()
    # test()

    # showImg(train_loader, 5)

    # 逐层测试网络的输出
    # for test_x, test_y in train_loader:
    #     test_x, test_y = test_x.to(device), test_y.to(device)
    #     out = net(test_x)[0]
    #     print(out,out.shape)
    #     break
    