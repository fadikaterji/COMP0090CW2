import IPython
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Module
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import os
import torchvision
import PIL

from PIL import Image
import torchvision.transforms.functional as TF
import h5py
import math
from torchviz import make_dot
import pickle
from networkaux import Net
from data import PetDataSet


def accuracy(DATALOADER):
  correct = 0
  total = 0

  for data in DATALOADER:
      images,_,labels = data
      images=images.to(device)
      labels=labels.to(device)
      outputs = net(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  print('Accuracy of the network : %.3f %%' % (
      100 * correct / total))
  return(100 * correct / total)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

"""## train the model"""

project_path='/mnt/c/Users/Fadik/COMP0090/cw2/cw2/'
DATAPATH_train=project_path+'datasets-oxpet/train'
DATAPATH_test=project_path+'datasets-oxpet/test'
train_dataset =PetDataSet(DATAPATH_train)
test_dataset =PetDataSet(DATAPATH_test)
print('loading data')
DATALOADER_train = torch.utils.data.DataLoader(train_dataset, shuffle=True,batch_size=32)
DATALOADER_test = torch.utils.data.DataLoader(test_dataset, shuffle=True,batch_size=32)
print('done')

net = Net().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-05)
num_epochs = 75
history = {'train_loss': []}


def train_epoch(model, device, dataloader, loss_fn, optimizer):
    # Set train mode for the model
    model.train()
    train_loss = []
    # Iterate the dataloader
    for image_batch, _, class_batch in dataloader:
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        class_batch = class_batch.to(device)
        class_batch = class_batch.type(torch.LongTensor)
        class_batch = class_batch.to(device)
        ###################################
        # predict
        outputs = model(image_batch)
        outputs = outputs.to(device)
        ########
        loss = loss_fn(outputs, class_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ##########
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


for epoch in range(num_epochs):
    train_loss = train_epoch(net, device, DATALOADER_train, criterion, optimizer)
    print('\n EPOCH {}/{} \t train loss {:.3f} '.format(epoch + 1, num_epochs, train_loss))
    history['train_loss'].append(train_loss)

torch.save(net.state_dict(), project_path + 'modelope.pth')  # save model

print("Training accuracy:")
train_acc=accuracy(DATALOADER_train)
print("\nTesting accuracy:")
test_acc=accuracy(DATALOADER_test)
