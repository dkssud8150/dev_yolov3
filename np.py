import torch
import torch.nn as nn

import numpy as np

import time
import sys
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm,trange

from function.convolution import Conv
from function.pool import Pool
from function.fc import FC
from function.activation import *

def convolution():
    print("convolution")

    # define the shape of input & weight
    in_w = 6
    in_h = 6
    in_c = 1
    out_c = 16
    batch = 1
    k_w = 3
    k_h = 3

    # define matrix
    x = np.arange(in_w*in_h*in_c*batch, dtype=np.float32).reshape([batch, in_c, in_h, in_w])
    w = np.array(np.random.standard_normal([out_c, in_c, k_h, k_w]), dtype=np.float32)
    #print(x,"\n\n", w)

    Convolution = Conv(batch = batch,
                        in_c = in_c,
                        out_c = out_c,
                        in_h = in_h,
                        in_w = in_w,
                        k_h = k_h,
                        k_w = k_w,
                        dilation = 1,
                        stride = 1,
                        pad = 0)

    print("x shape : ", x.shape)
    print("w shape : ", w.shape)

    l1_time = time.time()
    
    for i in range(100):
        L1 = Convolution.conv(x,w)
    print("L1 time : ", time.time() - l1_time)
    #print(L1)
    #print("L1 shape : ", L1.shape) # batch, out_c, out_h, out_w

    l2_time = time.time()

    for i in range(100):
        L2 = Convolution.gemm(x,w)
    print("L2 time : ", time.time() - l2_time)

    #print(L2)
    #print("L2 shape : ", L2.shape) # batch, out_c, out_h, out_w


    # pytorch
    torch_conv = nn.Conv2d(in_c,
                            out_c,
                            kernel_size = k_h,
                            stride = 1,
                            padding = 0,
                            bias = False,
                            dtype = torch.float32)
    torch_conv.weight = torch.nn.Parameter(torch.tensor(w))

    l3_time = time.time()
    
    for i in range(100):
        L3 = torch_conv(torch.tensor(x, requires_grad=False, dtype=torch.float32))
    print("L3 time : ", time.time() - l3_time)
    print(L3)


# 간단한 forward 구조 생성
def forward_net():
    # define
    batch = 1
    in_c = 3
    in_w = 6
    in_h = 6
    k_h = 3
    k_w = 3
    out_c = 1

    x = np.arange(batch*in_c*in_w*in_h, dtype=np.float32).reshape([batch, in_c, in_w, in_h])
    w1 = np.array(np.random.standard_normal([out_c, in_c, k_h, k_w]), dtype=np.float32)

    Convolution = Conv(batch = batch,
                    in_c = in_c,
                    out_c = out_c,
                    in_h = in_h,
                    in_w = in_w,
                    k_h = k_h,
                    k_w = k_w,
                    dilation = 1,
                    stride = 1,
                    pad = 0)

    L1 = Convolution.gemm(x,w1)
    print("L1 shape", L1.shape) # L1 shape (1, 1, 4, 4)
    print("L1", L1)

    Pooling = Pool(batch = batch, # L1의 출력 Shape를 입력으로 넣어줘야 한다.
                    in_c = L1.shape[1],
                    out_c = L1.shape[0],
                    in_h = L1.shape[2],
                    in_w = L1.shape[3],
                    kernel = 2, # pooling의 커널 2x2
                    dilation = 1,
                    stride = 2,
                    pad = 0)

    L1_max = Pooling.maxpool(L1)

    print("\nL1 max shape : ", L1_max.shape)
    print("L1 max",L1_max)


    # fully connected layer
    w2 = np.array(np.random.standard_normal([L1_max.shape[0], L1_max.shape[1]*L1_max.shape[2]*L1_max.shape[3]]), dtype=np.float32)
    Fc = FC(batch = L1_max.shape[0],
            in_c = L1_max.shape[1],
            out_c = 1, # 출력은 batch x 1 이어야 함
            in_h = L1_max.shape[2],
            in_w = L1_max.shape[3])

    L2 = Fc.fc(L1_max, w2)

    print("L2 shape : ", L2.shape)
    print(L2)

def plot_activation():
    x = np.arange(-10,10,1)
    out_relu = relu(x)
    out_leaky = leaky_relu(x)
    out_sigmoid = sigmoid(x)
    out_tanh = tanh(x)

    plt.figure(figsize=(10,5))
    output = {'out_relu':out_relu, 'out_leaky':out_leaky, 
                'out_sigmoid':out_sigmoid, 'out_tanh':out_tanh}
    key = list(output.keys())
    for i in range(len(key)):
        out = key[i]
        plt.subplot(2,2,i+1)
        plt.plot(x, output[out], 'r')
        plt.title(out)
        plt.tight_layout()

    plt.show()



def shallow_network():
    # input [1,1,6,6], 2 iter
    x = [np.array(np.random.standard_normal([1,1,6,6]), dtype=np.float32),
         np.array(np.random.standard_normal([1,1,6,6]), dtype=np.float32)]

    # Ground Truth
    y = np.array([1,1], dtype=np.float32)

    # conv1 weights [1,1,3,3]
    w1 = np.array(np.random.standard_normal([1,1,3,3]), dtype=np.float32)

    # fc weights [1,4]
    w2 = np.array(np.random.standard_normal([1,4]), dtype=np.float32)

    padding = 0
    stride = 1

    # L1 layer shape w,h
    L1_h = (x[0].shape[2] - w1.shape[2] + 2 * padding) // stride + 1
    L1_w = (x[0].shape[3] - w1.shape[3] + 2 * padding) // stride + 1

    #print("L1 output : ({}, {})".format(L1_h, L1_w)) # (4, 4)

    # conv1
    Convolution = Conv(batch = x[0].shape[0],
                       in_c = x[0].shape[1],
                       out_c = w1.shape[0],
                       in_h = x[0].shape[2],
                       in_w = x[0].shape[3],
                       k_h = w1.shape[2],
                       k_w = w1.shape[3],
                       dilation = 1,
                       stride = stride,
                       pad = padding)

    # conv1 backprop conv
    Conv_diff =  Conv(batch = x[0].shape[0],
                      in_c = x[0].shape[1],
                      out_c = w1.shape[0],
                      in_h = x[0].shape[2],
                      in_w = x[0].shape[3],
                      k_h = L1_h,
                      k_w = L1_w,
                      dilation = 1,
                      stride = stride,
                      pad = padding)

    # FC
    Fc = FC(batch = x[0].shape[0],
            in_c = x[0].shape[1],
            out_c = 1,
            in_h = L1_h,
            in_w = L1_w)

    # max pooling
    Pooling = Pool(batch = x[0].shape[0],
                   in_c = w1.shape[1],
                   out_c = w1.shape[0],
                   in_h = L1_h,
                   in_w = L1_w,
                   kernel = 2,
                   dilation = 1,
                   stride = 2,
                   pad = 0)

    epochs = 100

    for e in trange(epochs): # 100 epoch
        total_loss = 0
        for i in range(len(x)): # 2iter for each epoch
            # forward
            L1 = Convolution.gemm(x[i], w1)
            #print (x[i].shape, w1.shape, L1.shape) # (4, 4)
            
            L1_act = sigmoid(L1) # (1,1,4,4)
            #print (L1.shape, L1_act.shape)
            
            L1_max = Pooling.maxpool(L1_act)
            #print (L1_act.shape, L1_max.shape) # (1,1,2,2)
            
            L1_max_flatten = np.reshape(L1_max, (1,-1))
            #print (L1_max_flatten.shape) # (1,4)

            L2 = Fc.fc(L1_max_flatten, w2)
            #print (L2.shape) # (1,1)
            #print (L2)
            
            L2_act = sigmoid(L2)
            #print (L2_act.shape)


            loss = np.square(y[i] - L2_act) * 0.5
            total_loss += loss.item()
            #print (loss)


            # backward
            # delta E / delta w2
            diff_w2_1 = L2_act - y[i]

            diff_w2_2 = L2_act * ( 1 - L2_act)

            diff_w2_3 = L1_max

            diff_w2 = diff_w2_1 * diff_w2_2 * diff_w2_3
            #print (diff_w2.shape) # 2x2 인데, fc layer.shape은 1x4이므로 변환해줘야 함

            diff_w2 = np.reshape(diff_w2, (1,-1))


            # delta E / delta w1
            diff_w1_1 = diff_w2_1 * diff_w2_2
            #print (diff_w1_1.shape)
            diff_w1_2 = np.reshape(w2, (1,1,2,2)) # w2 [1,4] -> reshape
            #print (diff_w1_2.shape) # 1,1,2,2

            diff_w1_2 = diff_w1_2.repeat(2, axis=2).repeat(2, axis=3) # array를 n번 증폭
            #print (diff_w1_2.shape) # 1,1,4,4

            # diff maxpool
            diff_w1_3 = np.equal(L1_act, L1_max.repeat(2, axis=2).repeat(2, axis=3)) # pooling의 input, output,, 동일한 값의 인덱스를 구해줌, 동일한 행렬 크기로 만든 후 비교
            #print (diff_w1_3)

            diff_w1_4 = L1_act * (1- L1_act)
            #print (diff_w1_4.shape) # 1,1,4,4

            diff_w1 = diff_w1_1 * diff_w1_2 * diff_w1_3 * diff_w1_4 
            #print(diff_w1.shape)
            # 위 4개의 결과는 4x4 이고, x[i]는 6x6이므로 x[i]에 conv를 진행해줘야 함
            diff_w1 = Conv_diff.gemm(x[i], diff_w1)             

            # update
            w2 = w2 - 0.01 * diff_w2
            w1 = w1 - 0.01 * diff_w1

        print("{} epoch loss {}".format(e, total_loss / len(x)))



if __name__ == "__main__":
    #convolution()
    #forward_net()
    #plot_activation()
    shallow_network()