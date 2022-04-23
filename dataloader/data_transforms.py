import numpy as np
import cv2
import torch
import torchvision.transforms as transforms

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from util.tools import xywh2xyxy_np


def get_transformations(cfg_param = None, is_train = None):
    if is_train:
        data_transform = transforms.Compose([AbsoluteLabels(),
                                             DefaultAug(),
                                             RelativeLabels(),
                                             ResizeImage(new_size = (cfg_param['in_width'], cfg_param['in_height'])),
                                             ToTensor(),
                                            ])
    else:
        data_transform = transforms.Compose([AbsoluteLabels(),
                                             DefaultAug(),
                                             RelativeLabels(),
                                             ResizeImage(new_size = (cfg_param['in_width'], cfg_param['in_height'])),
                                             ToTensor(),
                                            ])

    return data_transform


# absolute bbox, 현재는 이미지 크기에 따른 0~1값을 가지지만, 이것을 절대값으로 가지고 있어야 transform을 해도 정보를 잃지 않는다.
class AbsoluteLabels(object):
    def __init__(self,):
        pass

    def __call__(self, data): # yolodata코드에서 transform이 들어갈 때 (img, bbox)가 data로 들어옴
        img, label = data
        h, w , _ = img.shape
        label[:,[1, 3]] *= w # cx, w *= w
        label[:,[2, 4]] *= h # cy, h *= h

        return img, label

# relative bbox
class RelativeLabels(object):
    def __init__(self,):
        pass

    def __call__(self, data):
        img, label = data
        h, w, _ = img.shape
        label[:,[1,3]] /= w
        label[:,[2,4]] /= h
        return img, label

class ResizeImage(object):
    def __init__(self, new_size, interpolation = cv2.INTER_LINEAR): # interpolation은 보간으로 이미지를 변환할 때 빈공간을 어떻게 처리할지
        self.new_size = tuple(new_size)
        self.interpolation = interpolation

    def __call__(self, data):
        img, label = data
        img = cv2.resize(img, self.new_size, interpolation=self.interpolation)
        return img, label
        # label은 normalize된 값이므로 resize하지 않아도 된다. 나중에 width, height를 곱하면 resize된 label로 만들어질 것이다.

class ToTensor(object):
    def __init__(self,):
        pass
    def __call__(self, data):
        img, label = data
        img = torch.tensor(np.transpose(np.array(img, dtype=float) / 255, (2,0,1)), dtype=torch.float32) # normalize, transpose HWC to CHW
        label = torch.FloatTensor(np.array(label))

        return img, label



# augmentation template, 앞으로 다른 augmentation을 사용할 때 이 template을 상속받아서 구현할 것이다. 공통적으로 augmentation을 할 때마다 bbox가 augmentation방식에 따라 값이 변해야 하므로
class ImgAug(object):
    def __init__(self, augmentations=[]):
        self.augmentations = augmentations
    
    def __call__(self, data):
        # unpack data
        img, labels = data
        # convert xywh to minx,miny,maxx,maxy because convenient and the imgAug format
        boxes = np.array(labels)
        boxes[:,1:] = xywh2xyxy_np(boxes[:,1:]) #0번째는 cls 정보이므로  

        # convert bbox to imgaug format
        bounding_boxes = BoundingBoxesOnImage(
                                [BoundingBox(*box[1:], label=box[0]) for box in boxes],
                                shape=img.shape)

        #apply augmentation
        img, bounding_boxes = self.augmentations(image=img,
                                                 bounding_boxes=bounding_boxes)

        # 예외 처리, 이미지 밖으로 나가는 bounding box를 제거
        bounding_boxes = bounding_boxes.clip_out_of_image()

        # convert bounding boxes to np.array()
        boxes = np.zeros((len(bounding_boxes), 5)) # memory assignment
        for box_idx, box in enumerate(bounding_boxes):
            x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2 # x1,y1,x2,y2 멤버 변수를 가지고 있음

            # return [x, y, w, h], 원래의 포맷은 xywh이므로 다시 변환
            boxes[box_idx, 0] = box.label
            boxes[box_idx, 1] = (x1 + x2) / 2
            boxes[box_idx, 2] = (y1 + y2) / 2
            boxes[box_idx, 3] = x2 - x1
            boxes[box_idx, 4] = y2 - y1

        return img, boxes


class DefaultAug(ImgAug):
    def __init__(self,):
        self.augmentations = iaa.Sequential([
                                    iaa.Sharpen(0.0, 0.1),
                                    iaa.Affine(rotate=(-0,0), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5))
                                    ])