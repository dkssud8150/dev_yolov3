# mnist classification
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim

import os, argparse, sys, time
from tqdm.notebook import trange

from model.models import *
from loss.loss import *
from util.tools import *
from train_test.train_and_eval import *
from train_test.test import *

def parse_args():
    parser = argparse.ArgumentParser(description="MNIST")
    parser.add_argument("--name", dest="dataset_name", help="what name do you use dataset",
                        default="mnist",type=str)
    parser.add_argument("--mode", dest="mode", help="train or valid or test", 
                        default=None, type=str)
    parser.add_argument("--download", dest="download", help="dataset download", 
                        default=False, type=bool)
    parser.add_argument("--odir", dest="output_dir", help="output directory for train result",
                        default="./output", type=str)
    parser.add_argument("--checkpoint",dest="checkpoint", help="checkpoint for trained model file", 
                        default=None, type=str)
    parser.add_argument("--device",dest="device", help="use device cpu / gpu",
                        default="cpu", type=str)
    parser.add_argument("--pretrain",dest="pretrain", help="when you use the pretrained model, True, or not False",
                        default=None, type=str)

    if len(sys.argv) == 1: # python main.py 하나만 했다는 뜻
        parser.print_help()
        sys.exit()
    
    args = parser.parse_args()
    return args


def main():
    #print(torch.__version__)

    if not os.path.isdir(args.output_dir + "/" + args.dataset_name + "/models/"):
        os.makedirs(args.output_dir + "/" + args.dataset_name + "/models/", exist_ok=True)
    
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            print("gpu")
            device = torch.device("cuda")
        else:
            print("can not use gpu!")
            device = torch.device("cpu")
    
    if args.download == True and args.dataset_name == 'kitti':
        os.system("./install_dataset.sh")

    if args.pretrain == "pretrain":
        model_name = 'pretrain'
    else:
        model_name = 'darknet'

    # MNIST 데이터 : [1, 32, 32], FashionMNIST 데이터 : [1, 28, 28]
    # [1, H, W]
    if args.mode == "train":
        train(args.dataset_name, args.output_dir, model_name=model_name, device=device, download=False)
    
    elif args.mode == "test":
        test(args.dataset_name, args.checkpoint, model_name=model_name, download=False, device=device)


if __name__ == "__main__":
    args = parse_args()
    main()
