import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleBlock(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(BottleBlock, self).__init__()

        self.bottleblock = nn.Sequential(
                                        nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=1, padding=0),
                                        nn.ReLU(),
                                        nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=1, padding=0),
                                        nn.ReLU(),
                                        nn.Conv2d(mid_dim, out_dim, kernel_size=1)

        )


    def forward(self ,x):
        return x

class Residual_Block(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(Residual_Block,self).__init__()
        # Residual Block
        self.residual_block = nn.Sequential(
                                            nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=1, padding=1), 
                                            nn.LeakyReLU(),
                                            nn.Conv2d(mid_dim, out_dim, kernel_size=3, stride=1, padding=0),
                                            )
        self.leakyrelu = nn.LeakyReLU()
                  
    def forward(self, x):
        out = self.residual_block(x)    # F(x)
        # print(out.shape, x.shape)
        out = out + x                   # F(x) + x
        out = self.leakyrelu(out)
        return out



class Darknet53(nn.Module):
    def __init__(self, batch, n_classes, in_channel=3, in_width=256, in_height=256, is_train = False):
        super(Darknet53, self).__init__()
        self.batch = batch
        self.n_classes = n_classes
        self.in_channel = in_channel
        self.in_width = in_width
        self.in_height = in_height
        self.is_train = is_train
        
        self.conv0 = nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=0)
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=0)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=0)
        self.conv6 = nn.Conv2d(1024, self.batch, kernel_size=1, stride=1, padding=0)

        self.rb0 = Residual_Block(in_dim=64, mid_dim=32, out_dim=64)
        self.rb1 = Residual_Block(in_dim=128, mid_dim=64, out_dim=128)
        self.rb2 = Residual_Block(in_dim=256, mid_dim=128, out_dim=256)
        self.rb3 = Residual_Block(in_dim=512, mid_dim=256, out_dim=512)
        self.rb4 = Residual_Block(in_dim=1024, mid_dim=512, out_dim=1024)



    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.rb0(x)

        x = self.conv2(x)
        x = self.rb1(x)

        x = self.conv3(x)
        x = self.rb2(x)

        x = self.conv4(x)
        x = self.rb3(x)

        x = self.conv5(x)
        x = self.rb4(x)

        x = F.adaptive_avg_pool2d(x, 1) # global avg pool
        x = self.conv6(x)
        # x = F.softmax(x, dim=1)             # softmax
        print(x.shape)

        if self.is_train == False:
            x = torch.argmax(x, dim=1)

        return x


''' 참고 블로그

global avg pooling : https://gaussian37.github.io/dl-concept-global_average_pooling/
residual block : https://coding-yoon.tistory.com/141

bottle block : https://coding-yoon.tistory.com/116?category=825914
'''
