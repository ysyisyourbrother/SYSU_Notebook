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

# 对图像进行变换，一般需要把PIL图像转成tensor，且归一化
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


# 定义训练数据集
## 该数据集是彩色图像，channel为3
train_data = torchvision.datasets.CIFAR10(
    root='/data/cifar10/cifar-10-python',
    train=True,
    transform=transform_train
)
# 使用Data的loader方便在训练时使用mini_batch
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


# 定义测试数据集
test_data = torchvision.datasets.CIFAR10(
    root='/data/cifar10/cifar-10-python', 
    train=False,
    transform=transform_test
)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)



'''
cifar10数据集，input size:(3,32,32)
'''
class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, 5, 1, 2), #(6,32,32)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) #(6,16,16)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6,12,3,1,1), #(12,16,16)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) #(12,,8,8)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(12,24,3,1,1), #(24,8,8)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) #(24,4,4)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(24,32,3,1,1), #(32,4,4)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), #(32,2,2)
        )

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.fc1 = nn.Linear(32*2*2, 10)
        self.fc2 = nn.Linear(56*4*4, 10)
        self.fc3 = nn.Linear(68*8*8, 10)
        self.fc4 = nn.Linear(74*16*16, 10)
        self.fc5 = nn.Linear(77*32*32, 10)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        # return x4
        x4_ = x4.view(x4.size(0), -1)
        out1 = self.fc1(x4_)
        # return out1

        x4 = self.upsample1(x4)
        x3 = torch.cat((x3,x4), 1)
        x3_ = x3.view(x3.size(0), -1)
        out2 = self.fc2(x3_)
        # return out2
        x3 = self.upsample1(x3)
        x2 = torch.cat((x2,x3), 1)
        x2_ = x2.view(x2.size(0), -1)
        out3 = self.fc3(x2_)

        x2 = self.upsample1(x2)
        x1 = torch.cat((x1,x2), 1)
        x1_ = x1.view(x1.size(0), -1)
        out4 = self.fc4(x1_)

        x1 = self.upsample1(x1)
        x = torch.cat((x,x1), 1)
        x_ = x.view(x.size(0), -1)
        out5 = self.fc5(x_)

        return out1, out2, out3, out4, out5



def train():
    net = NET().to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    best_accuracy = .0
    with open('unet.txt', 'w') as log_file:
        for epoch in range(EPOCHS):
            net.train()
            accu_loss = .0
            for step, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()

                out1, out2, out3, out4, out5 = net(batch_x)
                out = out1+out2+out3+out4+out5
                # print(out1, out1.shape)
                # break
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

                accu_loss += loss.item()
                if (step+1) % 200 == 0:   #每2000次batch_size的输入，打印一次结果
                        print('[%d, %5d] loss: %.3f' %(epoch + 1, step + 1, accu_loss / 2000)) #平均loss
                        log_file.write('[%d, %5d] loss: %.3f\n' %(epoch + 1, step + 1, accu_loss / 2000))
                        accu_loss = .0
            
            #测试
            net.eval()
            with torch.no_grad():
                correct = 0 # 记录整个测试集中预测正确的样本数量
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                    # output = net(batch_x)
                    out1, out2, out3, out4, out5 = net(batch_x)
                    out = out1+out2+out3+out4+out5
                    _, pred_label = torch.max(out, 1)
                    
                    correct += pred_label.eq(batch_y).sum().item()
                accuracy = correct / len(test_data) * 100.0
                print('EPOCH:%d: %.3f%%' %(epoch, accuracy))
                # log信息
                log_file.write('EPOCH:%d: %.3f%%\n' %(epoch, accuracy))
                
                # 若准确率提高了，则记录模型参数
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(net.state_dict(), 'unet.pth')
        log_file.write('Best Accuracy: %.3f\n' %best_accuracy)

if __name__ == "__main__":
    a = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float)
    b = torch.tensor([[1,1,1],[2,2,2]], dtype=torch.float)
    # c = torch.cat((a,b), 0)
    # print(a, a.shape)
    # print(b, b.shape)
    # print(c, c.shape)
    # net = NET().to(device)
    # c = torch.sum((a,b), 1)
    # c = a + b
    # print(c,c.shape)
    # c = c/2
    # print(c,c.shape)
    # mean = torch.mean(c,1)
    # print(mean)

    train()