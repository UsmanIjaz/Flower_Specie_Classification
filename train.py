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

from collections import OrderedDict
import copy
import time
import argparse

import helper as h

# defining the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default="/home/workspace/aipnd-project/flowers/" , type=str, help='data_dir describes path to the dataset ')
parser.add_argument('--gpu', action='store_true', help='gpu describes the use of GPU if available')
parser.add_argument('--epochs', default=5 ,type=int, help='epochs describes the number of epochs')
parser.add_argument('--arch', default="densenet" ,type=str, help='arch describes the model architecture')
parser.add_argument('--learning_rate', default=0.001 , type=float, help='learning_rate describes the learning rate for optimizer')
parser.add_argument('--hidden_units', default=512 , type=int, help='hidden_units describes the number of hidden units for classifier')
parser.add_argument('--save_dir', default="/home/workspace/aipnd-project/" ,type=str, help='save_dir describes the path to save trained model checkpoint file')
parser.add_argument('--checkpoint', default="checkpoint.pth", type=str, help='checkpoint describes the file name to save trained model checkpoint to file')

arguments = parser.parse_args()
print("Data Directory: ", arguments.data_dir)
print("GPU: ", arguments.gpu)
print("Epochs: ", arguments.epochs)
print("Model Architecture: ", arguments.arch)
print("Learning Rate: ", arguments.learning_rate)
print("Hidden Units: ", arguments.hidden_units)
print("Save Directory: ", arguments.save_dir)
print("Checkpoint File: ", arguments.checkpoint)
    



def main():
    if (arguments.arch != "densenet" and arguments.arch != 'alexnet'):
        print("Please, either choose densenet or alexnet as the model architecture, other models are not supported.")
        return 
    
    device = "cpu"
    if (arguments.gpu):
        if(torch.cuda.is_available()):
            device = "cuda"
            print(" GPU IS AVAILABLE")
        else:
            print("GPU is not avaliable, instead CPU will be used")



    h.train(arguments.data_dir, device, arguments.epochs, arguments.arch, arguments.learning_rate, arguments.hidden_units, arguments.save_dir,arguments.checkpoint )
    
if __name__ == "__main__":
    main()

