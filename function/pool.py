import numpy as np

# 2d pooling
class Pool:
    def __init__(self, batch, in_c, out_c, in_h, in_w, kernel, dilation, stride, pad):
        self.batch = batch
        self.in_c = in_c
        self.out_c = out_c
        self.in_h = in_h
        self.in_w = in_w
        self.kernel = kernel
        self.dilation = dilation
        self.stride = stride
        self.pad = pad

        self.out_w = (in_w - kernel + 2 * pad) // stride + 1
        self.out_h = (in_h - kernel + 2 * pad) // stride + 1

    def maxpool(self, A):
        C = np.zeros([self.batch, self.out_c, self.out_h, self.out_w], dtype=np.float32)
        for b in range(self.batch):
            for c in range(self.in_c):
                for oh in range(self.out_h): # output 크기만큼 결과를 낼 것이므로
                    a_j = oh * self.stride - self.pad # 연산 시작 row
                    for ow in range(self.out_w):
                        a_i = ow * self.stride - self.pad # 연산 col
                        # kernel 크기만큼 중에서 가장 큰 값을 지정
                        C[b, c, oh, ow] = np.amax(A[:, c, a_j:a_j+self.kernel, a_i:a_i+self.kernel])
        return C
    
    