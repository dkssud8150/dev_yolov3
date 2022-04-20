import torch
import torch.nn as nn

# in == input
class Lenet5(nn.Module):
    def __init__(self, batch, n_classes, in_channel, in_width, in_height, is_train = False):
        super().__init__()
        self.batch = batch
        self.n_classes = n_classes
        self.in_channel = in_channel
        self.in_width = in_width
        self.in_height = in_height
        self.is_train = is_train

        # layer define
        # convoluation output size : [(W - K + 2P)/S] + 1
        # w : input size, k : kernel size, p : padding size, s : stride
        self.conv0 = nn.Conv2d(self.in_channel, 6, kernel_size=5, stride=1, padding=0)
        # [(32 - 5 + 2*0) / 1] + 1 = 28
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)

        self.pool0 = nn.AvgPool2d(2, stride=2)
        self.pool1 = nn.AvgPool2d(2, stride=2)
        
        self.fc0 = nn.Linear(120, 84)
        self.fc1 = nn.Linear(84, self.n_classes)

        self.bn0 = nn.BatchNorm2d(6)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(120)

        self.dropout = nn.Dropout(p=0.3)

        self.leakyrelu = nn.LeakyReLU(0.1)


        # weight initialization
        torch.nn.init.xavier_uniform_(self.conv0.weight)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.fc0.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)


    # 실제 layer 연산 define    
    def forward(self, x):
        # x' shape = [B, C, H, W]
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.leakyrelu(x)
        x = self.pool0(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyrelu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leakyrelu(x)
        
        # change format from 3D to 2D ([B, C, H, W] -> B,C*H*W) 
        x = torch.flatten(x, start_dim=1)
        x = self.fc0(x)
        x = self.dropout(x)
        x = self.leakyrelu(x)
        x = self.fc1(x)
        x = x.view(self.batch, -1)
        x = nn.functional.softmax(x, dim=1) 
        
        # 학습할 때는 모든 결과를 받아야 함
        if self.is_train is False:
            x = torch.argmax(x, dim=1)

        return x


class my_Lenet5(nn.Module):
    def __init__(self, batch, n_classes, in_channel, in_width, in_height, is_train = False):
        super().__init__()
        self.batch = batch
        self.n_classes = n_classes
        self.in_channel = in_channel
        self.in_width = in_width
        self.in_height = in_height
        self.is_train = is_train

        # layer define
        # convoluation output size : [(W - K + 2P)/S] + 1
        # w : input size, k : kernel size, p : padding size, s : stride
        self.conv0 = nn.Conv2d(self.in_channel, 6, kernel_size=5, stride=1, padding=0)
        # [(32 - 5 + 2*0) / 1] + 1 = 28
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)

        self.pool0 = nn.MaxPool2d(2, stride=2)
        # self.pool0 = nn.AdaptiveAvgPool2d(2, stride=2)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        
        self.fc0 = nn.Linear(120, 84)
        self.fc1 = nn.Linear(84, self.n_classes)

        self.dropout = 0.5

        self.batch_norm1 = nn.BatchNorm1d(6)
        self.batch_norm2 = nn.BatchNorm1d(16)

    # 실제 layer 연산 define    
    def forward(self, x):
        # x' shape = [B, C, H, W]
        x = self.conv0(x)
        x = self.batch_norm1(x)
        x = torch.relu(x)
        x = self.pool0(x) # x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)

        x = self.conv1(x)
        x = self.batch_norm2(x)
        x = torch.relu(x)
        x = self.pool1(x) # x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        
        x = self.conv2(x)
        x = torch.relu(x)

        # change format from 3D to 2D ([B, C, H, W] -> B,C*H*W) 
        x = torch.flatten(x, start_dim=1)
        x = self.fc0(x)
        x = torch.relu(x)
        x = nn.functional.dropout(x, training=self.training, p = self.dropout)
        
        x = self.fc1(x)
        x = x.view(self.batch, -1)
        x = nn.functional.softmax(x, dim=1) 
        
        # 학습할 때는 모든 결과를 받아야 함
        if self.is_train is False:
            x = torch.argmax(x, dim=1)

        return x