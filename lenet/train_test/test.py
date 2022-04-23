import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim

from model.models import *
from loss.loss import *
from util.tools import *

import os, argparse, sys, time
from tqdm.notebook import trange


def test(dataset_name, checkpoint, model_name, download = False, device='cpu'):
    download_root = "./datasets"

    transform = {
        "test" : transforms.Compose([
            transforms.Resize([32,32]),
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(1.0,))
        ])
    }

    if dataset_name == "mnist":
        test_dataset = datasets.MNIST(root=download_root, 
                                transform=transform["test"], 
                                train=False,
                                download=download)

    elif dataset_name == "fashionmnist":
        test_dataset = datasets.FashionMNIST(root=download_root, 
                                transform=transform["test"], 
                                train=False,
                                download=download)


    test_dataloader = DataLoader(test_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True,
                                drop_last=False,
                                )

    # dataset 마다 이미지 크기가 다르므로
    if dataset_name == "mnist":
        n_classes, in_channel, in_height, in_width = 10, 1, 32, 32
    elif dataset_name == "fashionmnist":
        n_classes, in_channel, in_height, in_width = 10, 1, 28, 28
    elif dataset_name == "dogncat":
        n_classes, in_channel, in_height, in_width = 2, 3, 256, 256
    elif dataset_name == 'kitti':
        n_classes, in_channel, in_height, in_width = 10, 3, 256, 256
    

    _model = get_model(model_name)

    model = _model(batch=1, n_classes=n_classes, in_channel=in_channel, in_width=in_width, in_height=in_height)
    
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, (image, label) in enumerate(test_dataloader):
            image = image.to(device)
            output = model(image)
            output = output.cpu()
            #print(output)

            # show result
            # test의 경우 argmax를 수행한 결과를 리턴한다. 즉, label만 출력 output == prediction label
            show_img(image.detach().cpu().numpy(),str(output.item()))