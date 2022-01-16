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
from networkaux import Net

# setting the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

project_path='/mnt/c/Users/Fadik/COMP0090/cw2/cw2/'
net = Net()
net.load_state_dict(torch.load(project_path+'modelope.pth',map_location=torch.device(device.type)))

Encoder_cnn = net.features #use pre trained net.features layers as the encoder_cnn of reconstruction model


class Reconstruct(Module):

    def __init__(self, bottleneck_size):
        super().__init__()
        self.bottleneck_size = bottleneck_size

        self.flatten = nn.Flatten(start_dim=1)  # Flatten layer

        # One linear layer for encoder
        self.encoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(15 * 15 * 256, self.bottleneck_size),
            nn.ReLU(True)
        )

        # One linear layer for decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(self.bottleneck_size, 15 * 15 * 256),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 15, 15))  # Unflatten Layer

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
            nn.ConvTranspose2d(16, 3, 3, stride=1, padding=1),
        )

    def encoder(self, image):
        image = self.flatten(image)  # Flatten
        code = self.encoder_lin(image)  # Apply linear layers

        return code

    def decoder(self, code):
        code = self.decoder_lin(code)  # Apply linear layers
        code = self.unflatten(code)  # Unflatten
        decoded_image = self.decoder_conv(code)  # Apply transposed convolutions
        decoded_image = torch.sigmoid(decoded_image)  # Apply sigmoid activation

        return decoded_image

    def forward(self, image):
        code = self.encoder(image)
        decoded_image = self.decoder(code)

        return decoded_image
