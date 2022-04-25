import enum
import os, sys
import numpy as np
import torch
import torch.nn as nn

from util.tools import *

def make_conv_layer(layer_idx : int, modules : nn.Module, layer_info : dict, in_channels : int):
    filters = int(layer_info['filters']) # output channel size
    size = int(layer_info['size']) # kernel size
    stride = int(layer_info['stride'])
    pad = (size - 1) // 2 # layer_info['pad']
    modules.add_module('layer_'+str(layer_idx)+'_conv',
                        nn.Conv2d(in_channels, filters, size, stride, pad))

    if layer_info['batch_normalize'] == '1':
        modules.add_module('layer_'+str(layer_idx)+'_bn',
                        nn.BatchNorm2d(filters))

    if layer_info['activation'] == 'leaky':
        modules.add_module('layer_'+str(layer_idx)+'_act',
                        nn.LeakyReLU())
    elif layer_info['activation'] == 'relu':
        modules.add_module('layer_'+str(layer_idx)+'_act',
                        nn.ReLU())

def make_shortcut_layer(layer_idx : int, modules : nn.Module):
    modules.add_module('layer_'+str(layer_idx)+"_shortcut", nn.Identity()) # modulelist에서 info 타입이 맞지 않으면 복잡해지므로 빈 공간으로 init

def make_route_layer(layer_idx : int, modules : nn.Module):
    modules.add_module('layer_'+str(layer_idx)+"_route", nn.Identity())

def make_upsample_layer(layer_idx : int, modules : nn.Module, layer_info : dict):
    stride = int(layer_info['stride'])
    modules.add_module('layer_'+str(layer_idx)+'_upsample',
                        nn.Upsample(scale_factor=stride, mode='nearest'))


class Yololayer(nn.Module):
    def __init__(self, layer_info : dict, in_width : int, in_height : int, is_train : bool):
        super(Yololayer, self).__init__()
        self.n_classes = int(layer_info['classes'])
        self.ignore_thresh = float(layer_info['ignore_thresh'])  # loss 계산시 해당 박스가 특정 값 이상일 때만 연산에 포함되도록
        self.box_attr = self.n_classes + 5                     # output channel = box[4] + objectness[1] + class_prob[n]
        mask_idxes = [int(x) for x in layer_info['mask'].split(',')] # cfg파일에서 mask의 역할은 anchor가 총 9개 선언되어 잇는데, 각각의 yololayer에서 어떤 anchor를 사용할지에 대한 index이다. 0,1,2이면 0,1,2index의 anchor를 사용한다는 뜻
        anchor_all = [int(x) for x in layer_info['anchors'].split(',')] # w1,h1 , w2,h2 , w3,h3 , ... 로 되어 있으므로 이것을 다시 w,h 를 묶어줘야 한다.
        anchor_all = [(anchor_all[i], anchor_all[i+1]) for i in range(0, len(anchor_all), 2)]
        self.anchor = torch.tensor([anchor_all[x] for x in mask_idxes])
        self.in_width = in_width
        self.in_height = in_height
        self.stride = None # feature map의 1 grid가 차지하는 픽셀의 값 == n x n
        self.lw = None
        self.lh = None
        self.is_train = is_train

    def forward(self, x): # bounding box를 뽑을 수 있게 sigmoid나 exponantional을 취해줌
        # x is input. [N C H W]
        self.lw, self.lh = x.shape[3], x.shape[2] # feature map's width, height
        self.anchor = self.anchor.to(x.device) # 연산을 할 때 동일한 곳에 올라가 있어야함, cpu input이라면 cpu에, gpu input이라면 gpu에
        self.stride = torch.tensor([torch.div(self.in_width, self.lw, rounding_mode = 'floor'), 
                                    torch.div(self.in_height, self.lh, rounding_mode = 'floor')]).to(x.device) # stride = input size / feature map size
        
        # if kitti data, n_classes = 8, C = (8 + 5) * 3 = 39, yolo layer 이전의 filters 즉 output channels을 보면 다 39인 것을 확인할 수 있다.
        # [batch, box_attrib * anchor, lh, lw] ex) [1,39,19,19]
        # 4dim -> 5dim [batch, anchor, lh, lw, box_attrib]
        x = x.view(-1, self.anchor.shape[0], self.box_attr, self.lh, self.lw).permute(0,1,3,4,2).contiguous() # permute를 통해 dimension 순서를 변경, configuouse를 해야 바뀐채로 진행됨
        return x


class Darknet53(nn.Module):
    def __init__(self, cfg, param, training):
        super().__init__()
        self.batch = int(param['batch'])
        self.in_channels = int(param['in_channels'])
        self.in_width = int(param['in_width'])
        self.in_height = int(param['in_height'])
        self.n_classes = int(param['classes'])
        self.module_cfg = parse_model_config(cfg)
        self.module_list = self.set_layer(self.module_cfg)
        self.yolo_layers = [layer[0]for layer in self.module_list if isinstance(layer[0], Yololayer)]
        self.training = training

    def set_layer(self, layer_info): # init layer setting
        module_list = nn.ModuleList()
        in_channels = [self.in_channels] # first channels of input
        for layer_idx, info in enumerate(layer_info):
            modules = nn.Sequential()
            if info['type'] == 'convolutional':
                make_conv_layer(layer_idx, modules, info, in_channels[-1])
                in_channels.append(int(info['filters'])) # store each module's input channels
            elif info['type'] == 'shortcut':
                make_shortcut_layer(layer_idx, modules)
                in_channels.append(in_channels[-1])
            elif info['type'] == 'route':
                make_route_layer(layer_idx, modules)
                layers = [int(y) for y in info['layers'].split(',')]
                if len(layers) == 1:
                    in_channels.append(in_channels[layers[0]])
                elif len(layers) == 2:
                    in_channels.append(in_channels[layers[0]] + in_channels[layers[1]])
                
            elif info['type'] == 'upsample':
                make_upsample_layer(layer_idx, modules, info)
                in_channels.append(in_channels[-1]) # width, height만 커지므로 channel은 동일

            elif info['type'] == 'yolo':
                yololayer = Yololayer(info, self.in_width, self.in_height, self.training)
                modules.add_module('layer_'+str(layer_idx)+'_yolo', yololayer)
                in_channels.append(in_channels[-1])
            
            module_list.append(modules)
        return module_list

    def initialize_weights(self):
        # track all layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight) # weight initializing

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)  # scale
                nn.init.constant_(m.bias, 0)    # shift
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    


    def forward(self, x):
        yolo_result = [] # 최종 output, 마지막 layer가 yolo이므로
        layer_result = [] # shortcut, route에서 사용하기 위해 저장

        for idx, (name, layer) in enumerate(zip(self.module_cfg, self.module_list)):
            if name['type'] == 'convolutional':
                x = layer(x)
                layer_result.append(x)
            elif name['type'] == 'shortcut':
                x = x + layer_result[int(name['from'])]
                layer_result.append(x)
            elif name['type'] == 'yolo':
                yolo_x = layer(x)
                layer_result.append(yolo_x)
                yolo_result.append(yolo_x)
            elif name['type'] == 'upsample':
                x = layer(x)
                layer_result.append(x)
            elif name['type'] == 'route':
                layers = [int(y) for y in name['layers'].split(',')]
                x = torch.cat([layer_result[l] for l in layers], dim=1)
                layer_result.append(x)
            #print("idx : {}, result : {}".format(idx, layer_result[-1].shape))
        return yolo_result