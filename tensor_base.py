import torch
import numpy as np

def make_tensor():
    # int
    a = torch.tensor([[1, 2],[3, 4]], dtype=torch.int16)
    # float
    b = torch.tensor([1.0], dtype=torch.float32)
    # double
    c = torch.tensor([3], dtype=torch.float64, device="cuda:0")
    print(a.dtype, b.dtype, c.dtype, c.device)


def sumsub_tensor():
    a = torch.tensor([3,2])
    b = torch.tensor([5,3])
    
    print("input {}, {}\nsum : {}\tsub : {}".format(a,b, a+b, a-b))

    # each element sum
    sum_element_a = a.sum()
    print(sum_element_a)


def muldiv_tensor():
    a = torch.arange(0,9,1).view(3,3)
    b = torch.arange(0,9,1).view(3,3)

    # mat mul
    c = torch.matmul(a,b)
    print("input : {}\n{}\n\n mul : {}".format(a,b,c))

    # elementwisw multiplication
    d = torch.mul(a,b)
    print(d)



def reshape_tensor():
    a = torch.tensor([2,4,5,6,7,8])
    # view
    b = a.view(2,3)

    # transpose
    b_t = b.t()
    print("input : {} ,\t transpose : {}".format(b, b_t))


def access_tensor():
    a = torch.arange(1,13).view(4,3)
    
    # first row (slicing)
    print(a[:,0])

    # first col
    print(a[0,:])


def transform_numpy():
    a = torch.arange(1,13).view(4,3)

    # array to numpy
    a_np = a.numpy()
    print(a_np)

    # tensor to numpy
    b = np.array([1,2,3])
    b_ts = torch.from_numpy(b)
    print(b_ts)


# 두 dimension이 동일해야 한다.
def concat_tensor():
    a = torch.arange(1,10).view(3,3)
    b = torch.arange(10,19).view(3,3)
    c = torch.arange(19,28).view(3,3)

    abc = torch.cat([a,b,c],dim=1)
    print(abc)

    abc2 = torch.cat([a,b,c], dim=0)
    print(abc2)

    print(abc.shape, abc2.shape)


# 새로운 dimension을 만들면서 결합, 위는 2d -> 2d, 현재는 2d -> 3d
def stack_tensor():
    a = torch.arange(1,10).view(3,3)
    b = torch.arange(10,19).view(3,3)
    c = torch.arange(19,28).view(3,3)

    abc = torch.stack([a,b,c], dim=0) # 0index = a, 1index = b, 2index = c
    abc2 = torch.stack([a,b,c], dim=1) # 0index = a[0,:]+b[0,:]+c[0,:]
    abc3 = torch.stack([a,b,c], dim=2) # 0index = a[:,0]+b[:,0]+c[:,0]
    print(abc, '\n', abc2, "\n", abc3)
    print(abc.shape, abc2.shape)

def transpose_tensor():
    a = torch.arange(1,10).view(3,3)
    at = torch.transpose(a,0,1)
    print(a,"\n",at,"\n\n")

    b = torch.arange(1,25).view(4,3,2)
    bt = torch.transpose(b, 0, 2)
    print(b,"\n", bt)

    # 여러 가지 dimension을 바꿀 떄
    bp = b.permute(2,0,1) # 0,1,2 -> 2,0,1
    print(b.shape, bp.shape)



def quiz1():
    a = torch.arange(1,7).view(2,3)
    b = torch.arange(1,7).view(2,3)

    asumb = a+b
    asubb = a-b

    sum_element_a = a.sum()
    sum_element_b = b.sum()

    print(sum_element_a, sum_element_b)

def quiz2():
    a = torch.arange(1,46).view(1,5,3,3)

    trans = torch.transpose(a,1,3)

    print(trans[0,2,2,:])

def quiz3():
    a = torch.arange(1,7).view(2,3)
    b = torch.arange(1,7).view(2,3)
    c = torch.cat((a,b), dim=1)
    d = torch.stack((a,b),dim=0)

    print("concat : {}\nstack : {}".format(c.shape, d.shape))


if __name__ == '__main__':
    #make_tensor()
    #sumsub_tensor()
    #muldiv_tensor()
    #reshape_tensor()
    #access_tensor()
    #transform_numpy()
    #concat_tensor()
    #stack_tensor()
    #transpose_tensor()
    #quiz1()
    #quiz2()
    quiz3()