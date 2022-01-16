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
from networkaux2 import Net

# setting the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

#load trained model
project_path='/mnt/c/Users/Fadik/COMP0090/cw2/cw2/oeq/'
net = Net()

#net.load_state_dict(torch.load(project_path+'model_oeq_2.pth',map_location=torch.device(device.type)))
state_dict = torch.load(project_path+'model_oeq_2.pth',map_location=torch.device(device.type))
#Encoder=net.encoder_cnn #use pre trained net.features layers as the encoder of segmentation model
Encoder = state_dict['encoder_cnn'].to(device)
class Decoder(Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 5, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(32, 16, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1),
        )

    def forward(self, feature_image):
        # Apply transposed convolutions
        mask_image = self.decoder_conv(feature_image)
        # Apply a sigmoid
        mask = torch.sigmoid(mask_image)
        return mask


print(Encoder)

print(Decoder())
