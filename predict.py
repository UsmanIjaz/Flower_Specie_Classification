import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import torch.nn.functional as f
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
from PIL import Image

from collections import OrderedDict
import copy
import time
import argparse

import helper as h

# defining the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input', default="flowers/test/100/image_07899.jpg" , type=str, help='path to test image ')
parser.add_argument('--checkpoint', default='/home/workspace/aipnd-project/checkpoint_densenet.pth', help='path to checkpoint file')
parser.add_argument('--top_k', default=3 ,type=int, help='epochs describes the number of epochs')
parser.add_argument('--category_names', default="cat_to_name.json" ,type=str, help='path to file for categories name')
parser.add_argument('--gpu', action='store_true', help='gpu describes the use of GPU if available')

arguments = parser.parse_args()
print("Path to Input file: ", arguments.input)
print("Path to Checkpoint File: ", arguments.checkpoint)
print("Topk: ", arguments.top_k)
print("Path to Category file: ", arguments.category_names)
print("GPU: ", arguments.gpu)    



def main():
    
    device = "cpu"
    if (arguments.gpu):
        if(torch.cuda.is_available()):
            device = "cuda"
            print(" GPU IS AVAILABLE")
        else:
            print("GPU is not avaliable, instead CPU will be used")



    print(h.predict(arguments.input, arguments.checkpoint, arguments.top_k, arguments.category_names, device))
    
if __name__ == "__main__":
    main()

