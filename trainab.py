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
from data import PetDataSet
from networkab import Encoder
from networktarg import Decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
"""## train model"""
project_path='/mnt/c/Users/Fadik/COMP0090/cw2/cw2/'
DATAPATH_train=project_path+'datasets-oxpet/train'
DATAPATH_test=project_path+'datasets-oxpet/test'
train_dataset =PetDataSet(DATAPATH_train)
test_dataset =PetDataSet(DATAPATH_test)
DATALOADER_train = torch.utils.data.DataLoader(train_dataset, shuffle=True,batch_size=32)
DATALOADER_test = torch.utils.data.DataLoader(test_dataset, shuffle=True,batch_size=32)

loss_fn = torch.nn.MSELoss()
#loss_fn = torch.nn.BCELoss()
encoder_mtl= Encoder.to(device)
decoder_mtl= Decoder().to(device)

num_epochs = 150
lr= 0.0005# Learning rate
params_to_optimize = [
    {'params': encoder_mtl.parameters()},
    {'params': decoder_mtl.parameters()}
]
optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)
history_mtl={'train_loss':[]}
### Training function
def train_epoch(device, dataloader, loss_fn):
    # Set train mode for both the encoder and the decoder
    encoder_mtl.train()
    decoder_mtl.train()
    train_loss = []
    for image_batch,mask_batch,class_batch in dataloader:
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        mask_batch =mask_batch.to(device)
        ###
        encoded_data = encoder_mtl(image_batch)     # Encode data
        decoded_data = decoder_mtl(encoded_data)    # Decode data
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
   history_mtl['train_loss'].append(train_loss)

# save losses
with open(project_path+'loss_mtl2_oeq_ab.pkl', "wb") as tf:
    pickle.dump(history_mtl,tf)

torch.save({'encoder_mtl2':encoder_mtl, 'decoder_mtl2':decoder_mtl}, project_path+'model_mtl2.pth') # save model

state_dict = torch.load(project_path+'model_mtl2.pth',map_location=torch.device(device.type))
encoder_mtl2 = state_dict['encoder_mtl2'].to(device)
decoder_mtl2 = state_dict['decoder_mtl2'].to(device)

def accuracy_seg(segmented_image,target_image):
   acc=np.sum(segmented_image==target_image)/(segmented_image.shape[0]*segmented_image.shape[1])
   intersection = np.logical_and(target_image, segmented_image)
   union = np.logical_or(target_image,segmented_image)
   iou_score = np.sum(intersection) / np.sum(union)
   return (acc,iou_score)

for i in range(1,6):
  input_image=transforms.ToPILImage(mode='RGB')(train_dataset[20*i][0])
  target_image=np.array(np.array(transforms.ToPILImage(mode='L')(train_dataset[20*i][1]))>125,dtype='uint8')
  en=encoder_mtl2(train_dataset[20*i][0].unsqueeze(0).to(device))
  segmented_image=np.array(np.array(transforms.ToPILImage(mode='L')(decoder_mtl2(en)[0]))>125,dtype='uint8')
  '''
  fig, ax = plt.subplots(1,3)
  ax[0].imshow(input_image)
  ax[0].set_title('input_image')
  ax[1].imshow(segmented_image,cmap='gist_gray')
  ax[1].set_title('segmented_image')
  ax[2].imshow(target_image,cmap='gist_gray')
  ax[2].set_title('orginal_image2')
  plt.show()
  '''
acc_array_test=[]
ioc_array_test=[]
for i in range(len(test_dataset)):
  target_image=np.array(np.array(transforms.ToPILImage(mode='L')(test_dataset[i][1]))>125,dtype='uint8')
  en=encoder_mtl2(test_dataset[i][0].unsqueeze(0).to(device))
  segmented_image=np.array(np.array(transforms.ToPILImage(mode='L')(decoder_mtl2(en)[0]))>125,dtype='uint8')
  acc,ioc=accuracy_seg(segmented_image,target_image)
  acc_array_test.append(acc)
  ioc_array_test.append(ioc)

acc_array_train=[]
ioc_array_train=[]
for i in range(len(train_dataset)):
  target_image=np.array(np.array(transforms.ToPILImage(mode='L')(train_dataset[i][1]))>125,dtype='uint8')
  en=encoder_mtl2(train_dataset[i][0].unsqueeze(0).to(device))
  segmented_image=np.array(np.array(transforms.ToPILImage(mode='L')(decoder_mtl2(en)[0]))>125,dtype='uint8')
  acc,ioc=accuracy_seg(segmented_image,target_image)
  acc_array_train.append(acc)
  ioc_array_train.append(ioc)

mean_test_acc=np.mean(np.array(acc_array_test))
mean_test_ioc=np.mean(np.array(ioc_array_test))
mean_train_acc=np.mean(np.array(acc_array_train))
mean_train_ioc=np.mean(np.array(ioc_array_train))

print ('train accuracy : ',mean_train_acc)
print ('train IOC score : ',mean_train_ioc)
print ('test accuracy : ',mean_test_acc)
print ('test IOC score : ',mean_test_ioc)
