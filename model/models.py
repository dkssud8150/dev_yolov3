from model.lenet5 import Lenet5
from model.darknet import Darknet53

def get_model(model_name):
    if(model_name == "lenet5"):
        return Lenet5
    if(model_name == "darknet"):
        return Darknet53
    else:
        print("not exist this model : {}, download the pretrained model resnet50".format(model_name))







