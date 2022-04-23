import torch
from torch.utils.data import Dataset
import torchvision

from PIL import Image
import numpy as np

import os, sys

class Yolodata(Dataset): # torch utils data의 dataset을 상속받는다.

    # format path
    file_dir = ''
    anno_dir = ''
    file_txt = ''

    base_dir = 'C:\\Users\\dkssu\\dev\\datasets\\KITTI\\'
    # train dataset path
    train_img = base_dir + 'train\\JPEGimages\\'
    train_txt = base_dir + 'train\\Annotations\\'
    # valud dataset path
    valid_img = base_dir + 'valid\\JPEGimages\\'
    valid_txt = base_dir + 'valid\\Annotations\\'

    class_names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Persion_sitting', 'Cyclist', 'Tram', 'Misc'] # doncare는 x
    num_classes = None
    img_data = []

    def __init__(self, is_train=True, transform=None, cfg_param=None):
        super(Yolodata, self).__init__()
        self.is_train = is_train
        self.transform = transform
        self.num_class = cfg_param['classes']

        if self.is_train:
            self.file_dir = self.train_img
            self.anno_dir = self.train_txt
            self.file_txt = self.base_dir + 'train\\ImageSets\\train.txt'
        else:
            self.file_dir = self.valid_img
            self.anno_dir = self.valid_txt
            self.file_txt = self.base_dir + 'valid\\ImageSets\\valid.txt'

        img_names = []
        img_data = []

        with open(self.file_txt, 'r', encoding='UTF-8', errors='ignore') as f:
            img_names = [i.replace("\n", "") for i in f.readlines()]
            
        for i in img_names:
            if os.path.exists(self.file_dir + i + ".jpg"):
                img_data.append(i + ".jpg")
            elif os.path.exists(self.file_dir + i + ".JPG"):
                img_data.append(i + ".JPG")
            elif os.path.exists(self.file_dir + i + ".png"):
                img_data.append(i + ".png")
            elif os.path.exists(self.file_dir + i + ".PNG"):
                img_data.append(i + ".PNG")

        self.img_data = img_data
        print("data length : {}".format(len(self.img_data)))

    def __getitem__(self, index):
        img_path = self.file_dir + self.img_data[index]

        with open(img_path, 'rb') as f:
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
            img_origin_h, img_origin_w = img.shape[:2] # img shape : [H,W,C]

        # annotation dir이 있는지 확인, txt파일 읽기
        if os.path.isdir(self.anno_dir):
            txt_name = self.img_data[index]
            for ext in ['.png', '.PNG', '.jpg', '.JPG']:
                txt_name = txt_name.replace(ext, '.txt')
            anno_path = self.anno_dir + txt_name

            if not os.path.exists(anno_path):
                return
            
            bbox = []
            with open(anno_path, 'r') as f: # annotation about each image
                for line in f.readlines():
                    line = line.replace("\n",'')
                    gt_data = [l for l in line.split(' ')] # [class, center_x, center_y, width, height]

                    # skip when abnormal data
                    if len(gt_data) < 5:
                        continue

                    cls, cx, cy, w, h = float(gt_data[0]), float(gt_data[1]), float(gt_data[2]), float(gt_data[3]), float(gt_data[4])
                
                    bbox.append([cls, cx, cy, w, h])

            bbox = np.array(bbox)

            # skip empty target
            empty_target = False
            # even if target does not exist, we have to put bbox data
            if bbox.shape[0] == 0:
                empty_target = True
                # bbox의 형태가 객체가 2개일경우 [[a,b,c,d,e],[a,b,c,d,e]] 이므로 형태를 맞추기 위해 [[]]로 생성
                bbox = np.array([[0,0,0,0,0]])
            
            # data augmentation
            if self.transform is not None:
                img, bbox = self.transform((img, bbox))

            # 해당 배치가 몇번째 배치인지 확인하기 위한 index
            if not empty_target:
                batch_idx = torch.zeros(bbox.shape[0]) # 객체 개수만큼 크기를 생성
                target_data = torch.cat((batch_idx.view(-1,1), bbox), dim=1) # x는 1, y는 객체 개수의 array로 만들어줘서 bbox와 concat
            else:
                return
            return img, target_data, anno_path

        else: # test mode
            bbox = np.array([[0,0,0,0,0]])
            if self.transform is not None:
                img, _ = self.transform((img, bbox))
            return img, None, None

    def __len__(self):
        return len(self.img_data)