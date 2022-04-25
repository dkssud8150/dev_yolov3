import os, sys
import torch
import torch.optim as optim

from util.tools import *
from loss.loss import *

class Trainer:
    def __init__(self, model, train_loader, eval_loader, hyparam, device, torchwriter):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.max_batch = hyparam['max_batch']
        self.device = device
        self.epoch = 0
        self.iter = 0
        self.yololoss = Yololoss(self.device, self.model.n_classes)
        self.optimizer = optim.SGD(model.parameters(), lr=hyparam['lr'], momentum=hyparam['momentum'])

        self.scheduler_multistep = optim.lr_scheduler.MultiStepLR(self.optimizer, 
                                                             milestones=[20,40,60],
                                                             gamma = 0.5)            # 학습을 진행할 때마다 lr이 떨어져야 더 정교하게 학습이 가능하다. 떨어지는 빈도를 multisteplr로 설정
        self.torchwriter = torchwriter

    def run_iter(self):
        for i, batch in enumerate(self.train_loader):
            # drop the batch when invalid values
            if batch is None:
                continue

            input_img, targets, anno_path = batch
            #print("input {} {}".format(input_img.shape, targets.shape)) # [batch, C, H, W], [object number, (batchidx, cls_id, box_attirb)]
            input_img = input_img.to(self.device, non_blocking=True) # non_blocking

            output = self.model(input_img)

            # get loss between output and target(gt)
            loss, loss_list = self.yololoss.compute_loss(output, targets, self.model.yolo_layers)

            loss.backward()
            self.optimizer.step()       # gradient가 weight에 반영해서 update
            self.optimizer.zero_grad()  
            self.scheduler_multistep.step(self.iter) # step마다 lr을 줄임
            self.iter += 1

            # [total_loss.item(), lcls.item(), lobj.item(), lbox.item()]
            loss_name = ['total_loss', 'cls_loss', 'obj_loss', 'box_loss']

            if i % 10 == 0 :
                print("epoch {} / iter {} lr {} loss {}".format(self.epoch, self.iter, get_lr(self.optimizer), loss.item()))
                self.torchwriter.add_scalar('lr', get_lr(self.optimizer), self.iter)
                self.torchwriter.add_scalar('total_loss', loss, self.iter)
                for ln, lv in zip(loss_name, loss_list):
                    self.torchwriter.add_scalar(ln,lv,self.iter)

        return loss


    def run(self):
        while True:
            # train
            self.model.train()
            loss = self.run_iter()

            # save model 
            checkpoint_path = os.path.join("./output","model_epoch"+str(self.epoch)+".pth")
            torch.save({'epoch': self.epoch,
                        'iteration': self.iter,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss':loss}, 
                        checkpoint_path)
            
            # evaluation

            self.epoch += 1

            if self.epoch == self.max_batch:
                break

        



