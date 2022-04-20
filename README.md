# 가상환경 설정 (using virtualenv)


## 설치 및 가상환경 생성

```bash
pip install virtualenv

virtualenv dev --python=python3.8
```

이 때 3.8 interpreter가 없다고 뜬다면, 3.8 python이 안깔려 있는 것이므로 3.8버전을 깔거나 다른 버전으로 실행한다.

## 가상환경 활성화

```bash
source dev/Scripts/activate
```

## 필요한 패키지 설치

```bash
pip install numpy
```

## 나가기

```bash
deactivate
```

<br>

## 자신이 설치한 패키지를 저장하기

```bash
pip freeze > requirements.txt
```

## 다시 설치

```bash
pip install -r requirements.txt
```

<br>



<br>

FC 입력 시 maxpooling 하고 입력하는데, 왜 L1_h 그대로 사용하는가

```python
    Fc = FC(batch = x[0].shape[0],
            in_c = x[0].shape[1],
            out_c = 1,
            in_h = L1_h,
            in_w = L1_w)
```

1/2을 해주는 것이 맞지않나.

```python
Fc = FC(batch = x[0].shape[0],
        in_c = x[0].shape[1],
        out_c = 1,
        in_h = L1_h/2,
        in_w = L1_w/2)
```

FC 출력은 batch가 아닌 n_classes 개수로 출력이 되어야 하는 것이 아닌가

```python
import numpy as np

class FC:
    def __init__(self, batch, in_c, out_c, in_h, in_w):
        self.batch = batch
        self.in_c = in_c
        self.out_c = out_c
        self.in_h = in_h
        self.in_w = in_w

    def fc(self, A, W):
        # A shape : [b,in_c, in_h, in_w] -> [b, in_c*in_h*in_w]
        a_mat = A.reshape([self.batch, -1])
        B = np.dot(a_mat, np.transpose(W, (1,0))) # 이 때, W는 [b,in_c*in_h*in_w]이지만, 연산을 위해 [in_c*in_h*in_w, b]로 transpose
        return B
```

batch가 아니라 n_classes라는 표현이 맞지 않나

```python
import numpy as np

class FC:
    def __init__(self, n_classes, in_c, out_c, in_h, in_w):
        self.n_classes = n_classes
        self.in_c = in_c
        self.out_c = out_c
        self.in_h = in_h
        self.in_w = in_w

    def fc(self, A, W):
        # A shape : [b,in_c, in_h, in_w] -> [b, in_c*in_h*in_w]
        a_mat = A.reshape([self.n_classes, -1])
        B = np.dot(a_mat, np.transpose(W, (1,0))) # 이 때, W는 [b,in_c*in_h*in_w]이지만, 연산을 위해 [in_c*in_h*in_w, b]로 transpose
        return B
```

<br>

pooling을 진행할 때, weight의 차원은 [out_c, in_c, in_h, in_w]인데 왜 순서가 이러한가, in_c와 out_c가 바뀌어야 한다.

```python
    # max pooling
    Pooling = Pool(batch = x[0].shape[0],
                   in_c = w1.shape[0],
                   out_c = w1.shape[1],
                   in_h = L1_h,
                   in_w = L1_w,
                   kernel = 2,
                   dilation = 1,
                   stride = 2,
                   pad = 0)
```

```python
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
```


<br>

batchnorm, dropout, weight_decay 등을 사용하지 않았을 경우 정확도가 80%에서 막혔다.

batchNorm위치를 정확히 알고, weight initialization을 사용했을 때 정확도가 10프로정도 상승했다.

<br>

learning rate를 1e-3으로 주었더니 85%정도에 수렴된다. 이를 더 증가시키기 위해 lr = 1e-2로 바꾸었더니 아예 27%로 떨어졌다. 그래서 반대로 1e-4로 진행해보았다. 빠른 반복 내로 높게 올라가다 85에서 수렴되었다.


sgd로 변경시켜도 동일하게 84~85%에 계속 머물러있다. 이 경우 underfitting이라 생각하여 weight_decay를 증가시켜주었다. 81퍼에 머물럿다. adam의 weight decay를 1e-3으로 높여서 진행해보자.

