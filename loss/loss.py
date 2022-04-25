import torch
import torch.nn as nn
import sys
from util.tools import *

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



class Yololoss(nn.Module):
    def __init__(self, device, n_class):
        super(Yololoss, self).__init__()
        self.device = device
        self.n_class = n_class
        self.mseloss = nn.MSELoss().to(device) # mean squared entropy
        self.bceloss = nn.BCELoss().to(device) # binary cross entropy
        self.bcelogloss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device = device)).to(device) # take log for BCE
        
    def compute_loss(self, pred, targets, yololayer):
        lcls, lbox, lobj = torch.zeros(1, device = self.device), torch.zeros(1, device = self.device), torch.zeros(1, device = self.device) # loss_class, loss_box, loss_objectness
        
        tcls, tbox, tindices, tanchors = self.get_targets(pred, targets, yololayer)

        for pidx, pout in enumerate(pred):
            batch_id, anchor_id, gy, gx = tindices[pidx]
            # objectness information
            tobj = torch.zeros_like(pout[...,0], device=self.device)

            num_targets = batch_id.shape[0] # number of object in the batch size

            if num_targets:
                # pout shape : [batch, anchor, grid_h, grid_w, box_attrib]
                # get the only box_attrib information in grid, so then we can know batch index, anchor index
                ba = pout[batch_id, anchor_id, gy, gx]

                pred_xy = torch.sigmoid(ba[...,0:2]) 
                pred_wh = torch.exp(ba[...,2:4]) * tanchors[pidx]
                pred_box = torch.cat((pred_xy, pred_wh),dim=1) # pred_x,pred_y,pred_w,pred_h
                # print(pred_box.shape, tbox[pidx].shape)

                # iou
                iou = bbox_iou(pred_box, tbox[pidx], xyxy=False) # can get iou about each box 

                # box loss              
                lbox += (1 - iou).mean() # iou의 평균값들, 3 layer가 다 더해지도록

                # objectness loss
                # gt box and prediction box are coincide -> positive = 1, negative = 0
                # instead of dividing into 0 or 1, insert as a value between 0 and 1 
                tobj[batch_id, anchor_id, gy, gx] = iou.detach().clamp(0).type(tobj.dtype)


                # class loss
                if ba.size(1) - 5 > 1: # xywh, obj_info, cls_info 
                    t = torch.zeros_like(ba[...,5:], device = self.device)
                    # one hot encoding for the corresponding class
                    # if the information is for the 0th class, insert 1 into 0 index
                    t[range(num_targets), tcls[pidx]] = 1 
                    print(t)

                    # compute probability(ba[:,5:]) about class and list for 0 of non correct or 1 of correct (t) and sum
                    lcls += self.bcelogloss(ba[:,5:],t)

            # we can get also objectness loss, even if num_target is 0
            lobj += self.bcelogloss(pout[...,4], tobj)
                
        # assign loss weight, to set balence for each loss
        lcls *= 0.05
        lobj *= 1.0
        lbox *= 0.5

        total_loss = lcls + lbox + lobj
        
        # define the loss graph visualization
        loss_list = [total_loss.item(), lcls.item(), lobj.item(), lbox.item()]

        return total_loss, loss_list



    # for comparing prediction and gt conveniently, we transpose shape
    def get_targets(self, pred, targets, yololayer):
        num_anch = 3
        num_targets = targets.shape[0] # batch size
        tcls, tboxes, tindices, anch = [], [], [], [] # output, target_class, target_box, index, anchor

        gain = torch.ones(7, device=self.device) # targets is to be 7 dim, [b_id, c_id, cx,cy,w,h,a_id]

        # anchor index
        # ai.shape = (1x3) => 3x1, and repeat targets's num
        ai = torch.arange(num_anch, device=targets.device).float().view(num_anch, 1).repeat(1, num_targets)
        # to make targets to be anchor's number, targets.shape multiple anchor's num(3)

        targets = torch.cat((targets.repeat(num_anch, 1, 1), ai[:,:,None]), dim=2)


        for yi, yl in enumerate(yololayer):
            # 각 yolo layer feature map에 맞게 설정
            # cfg 파일에서의 anchors는 608에 대한 값, 19x19, 38x38에 대한 값으로 만들어줘야 함
            anchors = yl.anchor / yl.stride 
 
            gain[2:6] = torch.tensor(pred[yi].shape)[[3,2,3,2]] # [1,1,grid_w, grid_h, grid_w, grid_h,1]

            # multiple [box_cx, box_cy,box_w,box_y] * grid size, to unpack normalize
            t = targets * gain
            # print(t) # targets's[2:6] is to be some number dependent on grid size


            if num_targets:
                # in figure2 of yolov3 paper, w, h of bounding box is anchor size * exp(prediction's w) or exp(prediction's h)
                # so, r = exp(prediction_w) = box_w / anchor_w
                r = t[:,:,4:6] / anchors[:, None]

                # extract maximum exp(prediction_w)
                # select the ratios less than 4, remove the too large ratios
                # print(r)
                j = torch.max(r, 1. / r).max(dim = 2)[0] < 4
                # print("max : ", torch.max(r, 1. / r).max(dim = 2)[0])
                # print(j)

                t = t[j] # extract value for true
            else: # num_targets == 0
                t = targets[0]

            # batch_id, class_id with integer and transpose
            batch, cls = t[:,:2].long().T
            # print("batch, class", batch.shape, cls.shape, "\n", t[:, :2].shape, t.shape)

            gt_xy = t[:, 2:4]
            gt_wh = t[:, 4:6]

            # define the Cx, Cy in figure2. Cx Cy is index of grid
            # if in 19x19 gt_xy is 17.2,17.3, Cx Cy about object is 17,17
            gt_ij = gt_xy.long() # make integer from float type
            gt_i, gt_j = gt_ij.T # make 1 row, many col
            # print(gt_ij.shape, gt_i.shape, gt_j.shape)

            # anchor index
            a = t[:, 6].long()

            # add indices
            # clamp() : 19x19 이상의 값이 되지 않기 위해
            # always 0 < gt_j < grid_h -1 
            tindices.append((batch, a, gt_j.clamp(0, gain[3]-1), gt_i.clamp(0, gain[2]-1))) # [batch id, anchor id, Cy, Cx]

            # add target box
            # prediction_x, prediction_y normalized by sigmoid is box_x - Cx, or box_y - Cy in figure2   
            # shape : [p_x, p_y, gt_w, gt_h]
            tboxes.append(torch.cat((gt_xy-gt_ij, gt_wh), dim=1))

            # add anchor
            # a is index of anchor box to guess positive box, so insert anchor box for indices
            anch.append(anchors[a])

            # add class
            tcls.append(cls)

        return tcls, tboxes, tindices, anch