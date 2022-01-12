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
#import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
import h5py
import math
from torchviz import make_dot
import pickle
from networkbase import Encoder
###########Look at this################
from networktarg import Decoder
from data import PetDataSet

# setting the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

"""## train model"""

project_path='/mnt/c/Users/Fadik/COMP0090/cw2/'
DATAPATH_train=project_path+'datasets-oxpet/train'
DATAPATH_test=project_path+'datasets-oxpet/test'
train_dataset =PetDataSet(DATAPATH_train)
test_dataset =PetDataSet(DATAPATH_test)
DATALOADER_train = torch.utils.data.DataLoader(train_dataset, shuffle=True,batch_size=32)
DATALOADER_test = torch.utils.data.DataLoader(test_dataset, shuffle=True,batch_size=32)

loss_fn = torch.nn.MSELoss()
encoder_sgl= Encoder().to(device)
decoder_sgl= Decoder().to(device)

num_epochs = 10
lr= 0.0005# Learning rate
params_to_optimize = [
    {'params': encoder_sgl.parameters()},
    {'params': decoder_sgl.parameters()}
]
optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)
history_sgl={'train_loss':[]}
### Training function
def train_epoch(device, dataloader, loss_fn):
    # Set train mode for both the encoder and the decoder
    encoder_sgl.train()
    decoder_sgl.train()
    train_loss = []
    for image_batch,mask_batch,class_batch in dataloader:
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        mask_batch =mask_batch.to(device)
        ###
        encoded_data = encoder_sgl(image_batch)     # Encode data
        decoded_data = decoder_sgl(encoded_data)    # Decode data
        loss = loss_fn(decoded_data, mask_batch) # Evaluate loss
        optimizer.zero_grad() # Backward pass
        loss.backward()
        optimizer.step()
        print('\t partial train loss for single batch: %f' % (loss.data))# Print batch loss
        train_loss.append(loss.detach().cpu().numpy())
    ###display few results after every epoch
    ind=random.randint(0,1)
    org_image=transforms.ToPILImage(mode='L')(mask_batch[ind])
    pre_image=transforms.ToPILImage(mode='L')(decoded_data [ind])
    #fig, ax = plt.subplots(1,2)
    #ax[0].imshow(org_image,cmap='gist_gray')
    #ax[1].imshow(pre_image,cmap='gist_gray')
    #plt.show()
    return np.mean(train_loss)
for epoch in range(num_epochs):

   train_loss= train_epoch(device,DATALOADER_train,loss_fn)
   print('\n EPOCH {}/{} \t train loss1 {:.3f} '.format(epoch + 1, num_epochs,train_loss))
   history_sgl['train_loss'].append(train_loss)

#save losses
with open(project_path+'loss_sgl.pkl', "wb") as tf:
    pickle.dump(history_sgl,tf)

torch.save({'encoder_sgl':encoder_sgl, 'decoder_sgl':decoder_sgl}, project_path+'model_sgl.pth')#save model

state_dict = torch.load(project_path+'model_sgl.pth',map_location=torch.device(device.type))
encoder_sgl = state_dict['encoder_sgl'].to(device)
decoder_sgl = state_dict['decoder_sgl'].to(device)
