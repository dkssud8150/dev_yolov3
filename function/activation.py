import numpy as np

# max(0,x)
def relu(x):
    x_shape = x.shape
    x = np.reshape(x, [-1]) # 몇 차원인지 모르기 때문에 1차원으로 변환
    x = [max(0,v) for v in x]
    x = np.reshape(x, x_shape)
    return np.array(x, dtype=np.float32)

def leaky_relu(x):
    x_shape = x.shape
    x = np.reshape(x, [-1])
    x = [max(0.1*v, v) for v in x]
    x = np.reshape(x, x_shape)
    return np.array(x, dtype=np.float32)
    
def sigmoid(x):
    x_shape = x.shape
    x = np.reshape(x, [-1])
    x = [ 1 / (1 + np.exp(-v)) for v in x]
    x = np.reshape(x, x_shape)
    return np.array(x, dtype=np.float32)

def tanh(x):
    x_shape = x.shape
    x = np.reshape(x, [-1])
    x = [np.tanh(v) for v in x]
    x = np.reshape(x, x_shape)
    return np.array(x, dtype=np.float32)