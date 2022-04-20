import torch
import torch.nn as nn
import sys

class MNISTloss(nn.Module):
    def __init__(self, device = torch.device('cpu')):
        super(MNISTloss, self).__init__()
        self.loss = nn.CrossEntropyLoss().to(device)

    def forward(self, output, label):
        loss_val = self.loss(output, label)

        return loss_val

class FashionMNISTloss(nn.Module):
    def __init__(self, device = torch.device('cpu')):
        super(FashionMNISTloss, self).__init__()
        self.loss = nn.CrossEntropyLoss().to(device)
    
    def forward(self, output, label):
        loss_val = self.loss(output, label)

        return loss_val

def get_criterion(crit = "mnist", device = torch.device('cpu')):
    if crit == "mnist":
        return MNISTloss(device = device)
    elif crit == "fashionmnist":
        return FashionMNISTloss(device = device)
    elif crit == "dogncat":
        return FashionMNISTloss(device = device)
    else:
        print("unknown criterion")
        sys.exit(1)

