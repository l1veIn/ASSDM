#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import torch as t
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T


# In[3]:


transform = T.Compose([
    T.Resize(227),
    T.CenterCrop(227),
    T.ToTensor(),
    T.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
])


# In[56]:


class UCMdata(data.Dataset):
    dictionary = {
        'forest': 0,
        'buildings': 1,
        'river': 2,
        'mobilehomepark': 3,
        'harbor': 4,
        'golfcourse': 5,
        'agricultural': 6,
        'runway': 7,
        'baseballdiamond': 8,
        'overpass': 9,
        'chaparral': 10,
        'tenniscourt': 11,
        'intersection': 12,
        'airplane': 13,
        'parkinglot': 14,
        'sparseresidential': 15,
        'mediumresidential': 16,
        'denseresidential': 17,
        'beach': 18,
        'freeway': 19,
        'storagetanks': 20
    }

    def getFiles(self, root):
        files = os.listdir(root)
        for fi in files:
            fi_d = os.path.join(root, fi)
            if os.path.isdir(fi_d):
                self.getFiles(fi_d)
            else:
                self.imgs.append(fi_d)

    def __init__(self, root, transforms=transform, train=True, test=False ,start = 0., end =100.):
        self.train = train
        self.test = test
        self.imgs = []
        self.getFiles(root)
        self.imgs = sorted(self.imgs,
                           key=lambda x: x.split('.')[-2].split('/')[-1][-2:])
        imgs_num = len(self.imgs)    
        self.imgs = self.imgs[int(start * 0.01 * imgs_num):int(end * 0.01 * imgs_num)]
                  
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index]
#         label = t.zeros(21)
#         label[self.dictionary[img_path.split('/')[-2]]] = 1.0
        label = self.dictionary[img_path.split('/')[-2]]
        data = Image.open(img_path)
        if self.transforms:
            data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


# In[59]:


# newd = UCMdata('images')
# newd[0][1]

