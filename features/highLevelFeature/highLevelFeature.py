import torch as t
from tqdm import tqdm_notebook
import torch.nn as nn
import sys
sys.path.append("..")
sys.path.append("../..")
from data.dataset import UCMdata
from torch.utils.data import DataLoader
import os
current_path = os.path.dirname('__file__')
print(current_path)

class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        #remove last 2 layers
        self.fature = nn.Sequential(*list(model.children())[:-1])
        self.first_fc = nn.Sequential(*list(model.children())[-1][:1])
        self.second_fc = nn.Sequential(*list(model.children())[-1][1:3])

    def forward(self, x):
        x = self.fature(x)
        x = self.first_fc(x)
        x = x.view(1, -1)
        x = self.second_fc(x)
        return x

def highLevelFeature(path,numpy = True,modelPath=None):
    #load model
    image_data = UCMdata(path, example=True)
    image = DataLoader(image_data, 1, shuffle=True, num_workers=1)
    if not modelPath:
        AlexNet = t.load(current_path + 'test.pth')
    else:
        AlexNet = t.load(modelPath)
    model = Net(AlexNet) 
    return model(iter(image).next()[0]).squeeze().numpy()