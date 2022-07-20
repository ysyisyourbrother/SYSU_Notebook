import torch.nn as nn
import torch.nn.functional as F


# 定义网络结构
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 2),#input(3,32,32), output(16,32,32) (channel,h,w)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2) #output(16,16,16)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),#input(16,16,16), output(32,16,16)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2) #output(32,8,8)
        )

        self.linear1 = nn.Sequential(
            nn.Linear(32*8*8, 1024),
            nn.ReLU()
        )

        self.linear2 = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU()
        )

        self.linear3 = nn.Sequential(
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        # x = F.softmax(x, dim=1)
        # x = F.log_softmax(x, dim=1)
        return x


if __name__ == "__main__":
    net = CNN()
    print(net)