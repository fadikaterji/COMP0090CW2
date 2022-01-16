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
from data import PetDataSet
from networkaux import Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

project_path='/mnt/c/Users/Fadik/COMP0090/cw2/cw2/'
net = Net()
state_dict = torch.load(project_path+'model_oeq_2.pth',map_location=torch.device(device.type))
Encoder = state_dict['encoder_cnn'].to(device)

# Freeze weights
for param in Encoder.parameters():
    param.requires_grad = False
