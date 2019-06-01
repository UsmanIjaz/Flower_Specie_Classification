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
    
from PIL import Image
from collections import OrderedDict
import copy
import time

import json

def train(data_dir, available_device, num_epochs, arch, learning_rate, hidden_units, save_dir, checkpoint_file):

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(35),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])
    }

    batch_size = 32

    image_datasets = {
        'train' : ImageFolder(root= train_dir, transform=data_transforms['train']),
        'valid' : ImageFolder(root= valid_dir, transform=data_transforms['valid']),
        'test'  : ImageFolder(root= test_dir, transform=data_transforms['test'])
    }

    dataloaders =  {
        'train' : DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
        'valid' : DataLoader(image_datasets['valid'], batch_size=batch_size),
        'test'  : DataLoader(image_datasets['test'], batch_size=batch_size)
    }

    print("Size of training set is : "+str(len(image_datasets['train'])))
    print("Size of validation set is : "+str(len(image_datasets['valid'])))
    print("Size of testing set is : "+str(len(image_datasets['test'])))

    print("Number of classes are: " + str(len((image_datasets['train'].classes))))

    if arch == 'densenet':
        model = models.densenet121(pretrained=True)
        num_filters = 1024
        
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        num_filters = 9216

    device = torch.device(available_device)
    
    for param in model.parameters():
        param.requires_grad = False

    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(num_filters, hidden_units)),
        ('relu', nn.ReLU()),
        ('drpot', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    print_every = 60
    
    trainloader = dataloaders['train']
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    
    since = time.time()
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs = outputs.to(device)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() /len(image_datasets[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    checkpoint = {
              'epochs' : num_epochs,
              'optimizer_dict':optimizer.state_dict(),
              'class_to_idx': image_datasets['train'].class_to_idx,
              'state_dict': model.state_dict(),
              'model': arch,
              'hidden_units': hidden_units        
    }


    torch.save(checkpoint, save_dir+checkpoint_file)
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    loader = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])
    
    image_numpy = np.array(loader(Image.open(image)))
    
    return image_numpy

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['model'] == 'densenet':
        model = models.densenet121(pretrained=True)
        num_filters = 1024
    elif checkpoint['model'] == 'alexnet':
        model = models.alexnet(pretrained=True)    
        num_filters = 9216
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(num_filters, checkpoint['hidden_units'])),
                          ('relu', nn.ReLU()),
                          ('drpot', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(checkpoint['hidden_units'], 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = dict([[v,k] for k,v in model.class_to_idx.items()])
    
    
    return model

def predict(input_file,checkpoint, top_k, category_names, device):
    image = torch.FloatTensor([process_image(input_file)])
    model = load_checkpoint(checkpoint)
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    model.eval()
    output = model.forward(image)
    pobabilities = torch.exp(output).data.numpy()[0]
    
    top = np.argsort(pobabilities)[-top_k:][::-1] 
    top_probability = pobabilities[top]
    top_class = [model.idx_to_class[x] for x in top]
    top_class_names = [cat_to_name[y] for y in top_class]


    return top_probability, top_class,top_class_names