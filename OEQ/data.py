"""# Load Data"""
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

class PetDataSet(Dataset):

    def __init__(self, DATAPATH):
        super().__init__()

        img_file = DATAPATH + '/images.h5'
        msk_file = DATAPATH + '/masks.h5'
        bin_file = DATAPATH + '/binary.h5'
        img_h5 = h5py.File(img_file, 'r')
        msk_h5 = h5py.File(msk_file, 'r')
        bin_h5 = h5py.File(bin_file, 'r')

        self.images = np.array(img_h5.get('images'), dtype='uint8')
        self.masks = np.array(msk_h5.get('masks'), dtype='uint8')
        self.classes = np.array(bin_h5.get('binary'), dtype='uint8')

    def __len__(self):
        length = len(self.classes)

        return length

    def __getitem__(self, idx):
        image = self.images[idx]
        image = cv2.resize(image, (128, 128))  # resize the image
        mask = self.masks[idx] * 255  # conver binary mask into a normal image (0-255)
        mask = cv2.resize(mask, (128, 128))  # resie the msak image
        class_ = self.classes[idx][0]
        trn = transforms.Compose([
            transforms.ToTensor()])
        image = trn(image)
        mask = trn(mask)

        return (image, mask, class_)


