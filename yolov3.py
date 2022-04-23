import enum
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

import argparse

import os, sys, time
from dataloader.yolo_data import Yolodata

from util.tools import *
from dataloader.data_transforms import *
from model.yolov3 import * 
from train.train import * 

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOV3_PYTORCH arguments")
    parser.add_argument("--gpus", type=int, nargs='+', 
                        help="List of GPU device id", default=[])
    parser.add_argument("--mode", type=str, 
                        help="train / eval / test", default=None)
    parser.add_argument("--cfg", type=str,
                        help="model config path", default=None)
    parser.add_argument("--checkpoint", type=str,
                        help="model checkpoint path", default=None)
    parser.add_argument("--download", type=bool,
                        help="download the dataset", default=False)
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    args = parser.parse_args()
    return args


''' train '''
def train(cfg_param = None, using_gpus = None):
    print("train")
    
    my_transform = get_transformations(cfg_param=cfg_param, is_train=True)

    # dataloader
    train_dataset = Yolodata(is_train=True, 
                             transform=my_transform, 
                             cfg_param=cfg_param)
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=cfg_param['batch'],
                              num_workers = 0,       # num_worker : cpu와 gpu의 데이터 교류를 담당함. 0이면 default로 single process와 같이 진행, 0이상이면 multi thred
                              pin_memory = True,     # pin_memory : img나 데이터 array를 gpu로 올릴 때 memory의 위치를 고정시킨건지 할당할건지말지에 대한 것
                              drop_last = True,
                              shuffle = True)
                              #collate_fn=)          # collate_fn : batch size로 getitem할 때 각각의 이미지에 대해서만 가져온다. 그러나 학습을 할 떄는 batch 단위로 만들어줘야 하기 때문에 이를 collate fn으로 진행
    
    # # print EDA
    # for i, batch in enumerate(train_loader):
    #     img, target_data, anno_path = batch
    #     print("iter {}, img {}, targets {}, anno_path {}".format(i , img.shape, target_data.shape, anno_path))
    #     print("gt_data {}".format(target_data))
        
    #     drawBoxPIL(img[0].detach().cpu()) # detach는 backprop나 등등의 torch의 graph에서 떼내는 것이고, gpu이면 cpu로 데이터를 복사
    #     break

    model = Darknet53(args.cfg, cfg_param, training=True)

    model.train()
    model.initialize_weights()

    train = Trainer(model = model, train_loader = train_loader, eval_loader=None, hyparam=cfg_param)
    train.run()

 







''' eval '''
def eval(cfg_param = None, using_gpus = None):
    print("eval")

''' test '''
def test(cfg_param = None, using_gpus = None):
    print("test")


''' main '''
if __name__ == "__main__":
    args = parse_args()
    
    if args.download == True:
        os.system("./install_dataset.sh")

    # print config file
    net_data, conv_data = parse_hyperparam_config(args.cfg)
    
    cfg_param = get_hyperparam(net_data)


    usingf_gpus = [int(g) for g in args.gpus]

    if args.mode == "train":
        train(cfg_param = cfg_param)
    elif args.mode == "eval":
        eval(cfg_param = cfg_param)
    elif args.mode == "test":
        test(cfg_param = cfg_param)
    else:
        print("unknown mode")