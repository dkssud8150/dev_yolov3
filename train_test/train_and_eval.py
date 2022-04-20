import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from model.models import *
from loss.loss import *
from util.tools import *

import os, argparse, sys, time
from tqdm.notebook import trange
from glob import glob
import sklearn
from sklearn.model_selection import train_test_split

def train(dataset_name, output_dir, model_name, device="cpu", download=False):
    download_root = "./datasets"


    if dataset_name == "mnist":
        transform = {
            "train" : transforms.Compose([
                transforms.Resize([32,32]),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(1.0,))
            ]),
            "valid" : transforms.Compose([
                transforms.Resize([32,32]),
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(1.0,))
            ])
        }
        # MNIST(root, train=false,transform,...,download )
        train_dataset = datasets.MNIST(root=download_root,
                                transform=transform["train"],
                                train=True,
                                download=download)

        valid_dataset = datasets.MNIST(root=download_root, 
                                transform=transform["valid"], 
                                train=False,
                                download=download)
        
        class_names = train_dataset.classes


    elif dataset_name == "fashionmnist":
        transform = {
            "train" : transforms.Compose([
                transforms.Resize([32,32]),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(1.0,))
            ]),
            "valid" : transforms.Compose([
                transforms.Resize([32,32]),
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(1.0,))
            ])
        }
        train_dataset = datasets.FashionMNIST(root=download_root, 
                                transform=transform["train"], 
                                train=True,
                                download=download)
        valid_dataset = datasets.FashionMNIST(root=download_root, 
                                transform=transform["valid"], 
                                train=False,
                                download=download)
        
    
        class_names = train_dataset.classes

    elif dataset_name == "dogncat":
        transform = {
            "train" : transforms.Compose([
                transforms.Resize([244,244]),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(1.0,))
            ]),
            "valid" : transforms.Compose([
                transforms.Resize([244,244]),
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(1.0,))
            ])
        }

        train_dir = "./datasets/dogncat/train/"

        train_dataset = datasets.ImageFolder(train_dir, transform['train'])
        class_names = train_dataset.classes

        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [int(0.8 * len(train_dataset)), int(0.2 * len(train_dataset))])



    train_dataloader = DataLoader(train_dataset, 
                                    batch_size=8,
                                    shuffle=True,
                                    num_workers=0,
                                    pin_memory=True,
                                    drop_last=True,
                                    )
    valid_dataloader = DataLoader(valid_dataset,
                                    batch_size=8,
                                    shuffle=False,
                                    num_workers=0,
                                    pin_memory=True,
                                    drop_last=False,
                                    )

    if dataset_name == "mnist":
        n_classes, in_channel, in_height, in_width = 10, 1, 32, 32
    elif dataset_name == "fashionmnist":
        n_classes, in_channel, in_height, in_width = 10, 1, 28, 28
    elif dataset_name == "dogncat":
        n_classes, in_channel, in_height, in_width = 2, 3, 256, 256
    elif dataset_name == "kitti":
        n_classes, in_channel, in_height, in_width = 1, 1, 1, 1

    if model_name == 'lenet5' or model_name == 'darknet':
        _model = get_model(model_name)
        model = _model(batch=8, n_classes=n_classes, in_channel=in_channel, in_width=in_width, in_height=in_height, is_train=True)
        
    elif model_name == "resnet":
        model = models.resnet50(pretrained=True).to(device)
        num_features = model.fc.in_features

        model.fc = nn.Linear(num_features, n_classes)
    elif model_name == "fasterrcnn":
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)

    print("dataset name : {}".format(dataset_name))
    print("model name : {}".format(model_name))

    history = []
    model.to(device)
    model.train()
    
    # optimizer & scheduler
    # optimizer는 최적화 알고리즘, learning rate 는 learning rate 값, sgd 에서 얼마나 lr을 갱신할 지에 대한 momentum 
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    #optimizer = optim.SGD(model.parameters(), lr =1e-2, momentum=0.9, weight_decay=1e-3)

    # scheduler는 처음에는 0.01로 시작하지만, 너무 큰 lr이면 학습이 잘 안될 수 있으므로 줄여주기 위한 용도, steplr은 단계적으로 lr을 갱신하는 방법
    # step size 마다 gamma 값만큼 줄인다.
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    criterion = get_criterion(crit=dataset_name, device=device)

    epochs = 30
    
    best_loss = 5
    best_epoch = 0

    for epoch in range(epochs):
        start = time.time()

        total_loss = 0.0
        total_acc = 0.0

        
        for i, (image, label) in enumerate(train_dataloader):
            image = image.to(device)
            label = label.to(device)

            output = model(image)

            conf, preds = torch.max(output, dim=1)
            #print(output)

            loss_val = criterion(output, label)
            #print(loss_val)

            # backpropagation
            loss_val.backward()
            optimizer.step()
            optimizer.zero_grad()
            

            total_loss += loss_val.cpu().detach().numpy().item()
            total_acc += torch.sum(preds == label.data).cpu().detach().numpy()

            # if i % 1000 == 0:
            #     print("{} epoch {} iter loss : {}".format(epoch, i, loss_val.item()))

        total_loss = total_loss / len(train_dataset)
        total_acc = total_acc / len(train_dataset)
        scheduler.step()

        #if epoch+1 % 10 == 0:
        print("{} epoch \tloss : {:.4f}\tAcc : {:.4f}%\ttime : {:.2f}s".format(epoch+1, total_loss, total_acc*100, time.time() - start))

        history.append([total_loss, total_acc])

        if total_loss < best_loss:
            best_loss = total_loss
            best_epoch = epoch
            
    if epoch >= 15:
        torch.save(model.state_dict(), output_dir + "/" + dataset_name + "/models/" + "best_epoch.pt")

    result_plot(history)


    # eval
    model.eval()

    with torch.no_grad():
        total_acc = 0

        for i, (image, label) in enumerate(valid_dataloader):
            image = image.to(device)
            output = model(image)
            output = output.cpu()
            _, preds = torch.max(output, dim=1)
            #print(output)

            total_acc += torch.sum(preds == label.data).cpu().detach().numpy()

        total_acc = total_acc / len(train_dataset)

        print("pred : {}, Acc : {:.4f}%".format(class_names[preds[0]],total_acc * 100))
