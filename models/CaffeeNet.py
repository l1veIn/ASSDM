#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
from .BasicModule import BasicModule
import torch.nn as nn
import torch.nn.functional as F


class CaffeeNet(BasicModule):
    def __init__(self):
        super(CaffeeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11,1,3)
        self.conv2 = nn.Conv2d(96, 256, 5,1,2)
        self.conv3 = nn.Conv2d(256, 384, 3,1,1)
        self.conv4 = nn.Conv2d(384, 384, 3,1,1)
        self.conv5 = nn.Conv2d(384, 256, 3,1,1)
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 21)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (4, 4))
#         print("conv1:",x.size())
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
#         print("conv2:",x.size())
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
#         print("conv3:",x.size())
        x = F.relu(self.conv4(x))
#         print("conv4:",x.size())
        x = F.max_pool2d(F.relu(self.conv5(x)),(3,3))
#         print("conv5:",x.size())
        x = x.view(x.size()[0], -1)
#         print("reshape:",x.size())
        x = F.relu(self.fc1(x))
#         print("fc1:",x.size())
        x = F.relu(self.fc2(x))
#         print("fc2:",x.size())
        x = F.softmax(F.relu(self.fc3(x)),dim=1)
#         print("softMax:",x.size())
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# net = Net()
# print(net)


# In[8]:


# input = torch.autograd.Variable(torch.randn(1,3,227,227))
# print(input)
# out = net(input)
# print(out)

