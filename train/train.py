import os, sys
import torch
import torch.optim as optim

from util.tools import *

class Trainer:
    def __init__(self, model, train_loader, eval_loader, hyparam):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.max_batch = hyparam['max_batch']
        self.epoch = 0
        self.iter = 0
        self.optimizer = optim.SGD(model.parameters(), lr=hyparam['lr'], momentum=hyparam['momentum'])

        scheduler_multistep = optim.lr_scheduler.MultiStepLR(self.optimizer, 
                                                             milestones=[20,40,60],
                                                             gamma = 0.5)            # 학습을 진행할 때마다 lr이 떨어져야 더 정교하게 학습이 가능하다. 떨어지는 빈도를 multisteplr로 설정

    def run_iter(self):
        for i, batch in enumerate(self.train_loader):
            # drop the batch when invalid values
            if batch is None:
                continue

            input_img, targets, anno_path = batch
            print(input_img.shape, targets.shape)


    def run(self):
        while True:
            self.model.train()
            self.run_iter()
            
            # loss calculation
            

            self.epoch += 1





