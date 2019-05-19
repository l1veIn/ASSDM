import torch as t
from tqdm import tqdm_notebook
import torch.nn as nn
import sys
sys.path.append("..")
sys.path.append("../..")
from data.dataset import UCMdata

class Net(nn.Module):
    def __init__(self , model):
        super(Net, self).__init__()
        #remove last 2 layers
        self.fature = nn.Sequential(*list(model.children())[:-1])
        self.first_fc = nn.Sequential(*list(model.children())[-1][:1])
        self.second_fc = nn.Sequential(*list(model.children())[-1][1:3])
    def forward(self, x):
        x = self.fature(x)
#         print(x.size())
        x = self.first_fc(x)
#         print(x.size())
        x = x.view(1,-1)
#         print(x.size())
        x = self.second_fc(x)
        
        
        return x

def highLevelFeature(image):
    #load model
    AlexNet = t.load('../../checkpoints/0509_21:33:01.pth')
    model=Net(AlexNet)
#     print(list(model.children()))
    return model(image).squeeze()