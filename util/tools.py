from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

def show_img(img_data, text):
    # 입력된 값이 0~1로 정규화된 값이므로 255를 곱함
    _img_data = img_data * 255
    
    # 4d tensor -> 2d
    _img_data = np.array(_img_data[0,0], dtype=np.uint8)

    img_data = Image.fromarray(_img_data)
    draw = ImageDraw.Draw(img_data)

    # 예측결과을 텍스트로 보기 위해 선언
    # draw text in img, center_x, center_y
    cx, cy = _img_data.shape[0] / 2, _img_data.shape[1] / 2
    if text is not None:
        draw.text((cx,cy), text)

    plt.imshow(img_data)
    plt.show()

def result_subplot(epochs, history):
    history = np.array(history)
    loss = history[0][0]
    acc = history[0][1]
    x = np.linspace(0, epochs, epochs)
    out = [loss, acc]
    for i in range(2):
        plt.subplot(1,2,i+1)
        plt.plot(x, out[i])
        plt.title('training' + str(out[i]))
        plt.legend()
        plt.tight_layout()
    plt.show()

def result_plot(history):
    history = np.array(history)
    plt.plot(history[:,0:2])
    plt.text(0,0, "max accuracy" + str(max(history[:,0])))
    plt.legend(['loss', 'accuracy'])
    plt.show()


# parse model layer configuration
def parse_model_config(path):
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]

    module_defs = []
    type_name = None
    for line in lines:
        if line.startswith("["):
            type_name = line[1:-1].rstrip()
            if type_name == 'net':
                continue
            module_defs.append({})
            module_defs[-1]['type'] = type_name
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0

        else:
            if type_name == "net":
                continue
            
            key, value = line.split('=')
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs


# watch parse the yolov3 configuaraion about network
def parse_hyperparam_config(path):
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')] # #으로 시작하지 않는 줄만 저장
    lines = [x.rstrip().lstrip() for x in lines]

    # network hyperparameter에 대한 definition
    module_defs = []

    # convolution에 대한 parameter
    conv_defs = []

    for line in lines:
        # layer devision
        if line.startswith("["):
            type_name = line[1:-1].rstrip()
            if type_name == "net": # net은 network의 hyperparameter
                module_defs.append({}) # dictionary
                module_defs[-1]['type'] = type_name
            if type_name == 'convolutional':
                conv_defs.append({})
                conv_defs[-1]['type'] = type_name
        
        else:
            key, value = line.split("=")
            if type_name == 'net':
                module_defs[-1][key.rstrip()] = value.strip()
            if type_name == 'convolutional':
                conv_defs[-1][key.rstrip()] = value.strip()

    return module_defs, conv_defs

# get the data to want to be ours.
def get_hyperparam(data):
    for d in data:
        if d['type'] == 'net':
            batch = int(d['batch'])
            subdivision = int(d['subdivisions'])
            momentum = float(d['momentum'])
            decay = float(d['decay'])
            saturation = float(d['saturation'])
            lr = float(d['learning_rate'])
            burn_in = int(d['burn_in'])
            max_batch = int(d['max_batches'])
            lr_policy = d['policy']
            in_width = int(d['width'])
            in_height = int(d['height'])
            in_channels = int(d['channels'])
            classes = int(d['class'])
            ignore_class = int(d['ignore_cls'])

            return{'batch': batch,
                   'subdivision': subdivision,
                   'momentum': momentum,
                   'decay': decay,
                   'saturation': saturation,
                   'lr': lr,
                   'burn_in': burn_in,
                   'max_batch': max_batch,
                   'lr_policy': lr_policy,
                   'in_width': in_width,
                   'in_height': in_height,
                   'in_channels': in_channels,
                   'classes': classes,
                   'ignore_class': ignore_class}
            
        else:
            continue


def xywh2xyxy_np(x:np.array):
    y = np.zeros_like(x)
    y[...,0] = x[...,0] - x[...,2] / 2 # centerx - w/2 = minx
    y[...,1] = x[...,1] - x[...,3] / 2 # miny
    y[...,2] = x[...,0] + x[...,2] / 2 # maxx
    y[...,3] = x[...,1] + x[...,3] / 2 # maxy

    return y


def drawBoxPIL(img):
    img = img * 255

    if img.shape[0] == 3:
        img_data = np.array(np.transpose(img, (1,2,0)), dtype=np.uint8)
        img_data = Image.fromarray(img_data)

    # draw = ImageDraw.Draw(img_data)
    plt.imshow(img_data)
    plt.show()


import cv2
def drawBoxCV(img):
    img = img * 255

    if img.shape[0] == 3:
        img_data = np.array(np.transpose(img, (1,2,0)), dtype=np.uint8)

        cv2.imshow("img", img_data)
        cv2.waitKey()