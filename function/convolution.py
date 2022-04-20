import numpy as np

class Conv:
    # dilation : kernel이 얼마나 간격을 띄우고 연산할지에 대한 값
    def __init__(self, batch, in_c, out_c, in_h, in_w, k_h, k_w, dilation, stride, pad):
        self.batch = batch
        self.in_c = in_c
        self.out_c = out_c
        self.in_h = in_h
        self.in_w = in_w
        self.k_h = k_h
        self.k_w = k_w
        self.dilation = dilation
        self.stride = stride
        self.pad = pad

        self.out_h = (in_h - k_h + 2 * pad) // stride + 1
        self.out_w = (in_w - k_w + 2 * pad) // stride + 1

    def check_out(self, a, b):
        return a > -1 and a < b

    # naive convolution, sliding window matric
    def conv(self, A, B):

        # A * B = C
        # defice C size
        C = np.zeros((self.batch, self.out_c, self.out_h, self.out_w), dtype=np.float32)

        # 7 loop
        for b in range(self.batch):
            for oc in range(self.out_c):
                # each channel of output
                for oh in range(self.out_h):
                    for ow in range(self.out_w):
                        # each pixel of output shape
                        a_j = oh * self.stride - self.pad # a's y value
                        for kh in range(self.k_h):
                            if self.check_out(a_j, self.in_h) == False: # a_j 가 in_h보다 크다면 연산 x
                                C[b, oc, oh, ow] += 0
                            else:
                                a_i = ow * self.stride - self.pad # a's x value
                                for kw in range(self.k_w):
                                    if self.check_out(a_i, self.in_w) == False:
                                        C[b, oc, oh, ow] += 0
                                    else:
                                        C[b, oc, oh, ow] += np.dot(A[b, :, a_j, a_i], B[oc, :, kh, kw])
                                    a_i += self.stride # add x direction moving unit for kernel 
                            a_j += self.stride # add y direction moving unit for kernel
        return C

    # IM2COL, change n-dim input to 2-dim matrix
    def im2col(self, A):

        # define output as 2-dim matrix
        mat = np.zeros((self.in_c * self.k_h * self.k_w, self.out_w * self.out_h), dtype=np.float32)

        # matrix index
        mat_i = 0
        mat_j = 0

        # transform from A to mat
        for c in range(self.in_c):
            for kh in range(self.k_h):
                for kw in range(self.k_w):
                    in_j = kh * self.dilation - self.pad
                    for oh in range(self.out_h):
                        if not self.check_out(in_j, self.in_h):
                            for ow in range(self.out_w):
                                mat[mat_j, mat_i] = 0
                                mat_i += 1
                        else:
                            in_i = kw * self.dilation - self.pad
                            for ow in range(self.out_w):
                                if not self.check_out(in_i, self.in_w):
                                    mat[mat_j, mat_i] = 0
                                    mat_i += 1
                                else:
                                    mat[mat_j, mat_i] = A[0, c, in_j, in_i]
                                    mat_i += 1
                                in_i += self.stride
                        in_j += self.stride
                    mat_i = 0
                    mat_j += 1

        return mat
    
    # gemm, 2D matrix multiplication
    def gemm(self, A, B):
        a_mat = self.im2col(A)
        b_mat = B.reshape(B.shape[0],-1) 
        c_mat = np.matmul(b_mat, a_mat)

        c = c_mat.reshape([self.batch, self.out_c, self.out_h, self.out_w])
        return c