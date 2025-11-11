import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor
import os
import logging
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


class ResidualAdd(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return  x + self.f(x)
    
class EEGProjectLayer(nn.Module):
    def __init__(self,  z_dim,c_num, timesteps, drop_proj=0.3):
        super(EEGProjectLayer, self).__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps

        self.input_dim = self.c_num * (self.timesteps[1]-self.timesteps[0])
        proj_dim = z_dim

        self.model = nn.Sequential(nn.Linear(self.input_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim)
            )

        self.logit_scale = nn.Parameter(torch.tensor(100).log())
        self.softplus = nn.Softplus()
        self.eeg_alpha = nn.Parameter(torch.tensor(z_dim**-0.5).log())
        self.visual_alpha = nn.Parameter(torch.tensor(z_dim**-0.5).log())    
        self.curv = nn.Parameter(torch.tensor(1).log(), requires_grad=True)

        self.image_layer1 = nn.Linear(proj_dim, proj_dim)
        self.image_layer2 = nn.Linear(proj_dim, proj_dim)
        self.score_layer = nn.Linear(proj_dim, 1)
        

    def image_projection1(self, x):
        return self.image_layer1(x)

    def image_projection2(self, x):
        return self.image_layer2(x)
    
    def score(self, x):
        x = self.score_layer(x)
        return torch.sigmoid(x)
    
    def forward(self, x):
        x = x.view(x.shape[0], self.input_dim)
        x = self.model(x)
        return x



class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x
    
class BaseModel(nn.Module):
    # def __init__(self,  z_dim, c_num, timesteps, embedding_dim = 1040): # for MEG
    def __init__(self,  z_dim, c_num, timesteps, embedding_dim = 1440): # for eeg
        super(BaseModel, self).__init__()

        self.backbone = None
        self.project = nn.Sequential(
            FlattenHead(),
            nn.Linear(embedding_dim, z_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(z_dim, z_dim),
                nn.Dropout(0.5))),
            nn.LayerNorm(z_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()
        self.curv = nn.Parameter(torch.tensor(1).log(), requires_grad=True)
        self.eeg_alpha = nn.Parameter(torch.tensor(z_dim**-0.5).log())  
        self.visual_alpha = nn.Parameter(torch.tensor(z_dim**-0.5).log())    

        self.image_layer1 = nn.Linear(z_dim, z_dim)
        self.image_layer2 = nn.Linear(z_dim, z_dim)
        self.score_layer = nn.Linear(z_dim, 1)


    def score(self, x):
        x = self.score_layer(x)
        return torch.sigmoid(x)
    
    def image_projection1(self, x):
        return self.image_layer1(x)

    def image_projection2(self, x):
        return self.image_layer2(x)
    
    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.backbone(x)
        x = self.project(x)
        return x

class Shallownet(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps)
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 40, (1, 25), (1, 1)),
                nn.Conv2d(40, 40, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.AvgPool2d((1, 51), (1, 5)),
                nn.Dropout(0.5),
            )
    
class Deepnet(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps,embedding_dim = 1400)
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 25, (1, 10), (1, 1)),
                nn.Conv2d(25, 25, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(25),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),

                nn.Conv2d(25, 50, (1, 10), (1, 1)),
                nn.BatchNorm2d(50),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),

                nn.Conv2d(50, 100, (1, 10), (1, 1)),
                nn.BatchNorm2d(100),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),

                nn.Conv2d(100, 200, (1, 10), (1, 1)),
                nn.BatchNorm2d(200),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),
            )
        
class EEGnet(BaseModel):
    def __init__(self,  z_dim, c_num, timesteps):
        # super().__init__(z_dim, c_num, timesteps, embedding_dim = 864) for MEG
        super().__init__(z_dim, c_num, timesteps, embedding_dim = 1248) # for EEG
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 8, (1, 64), (1, 1)),
                nn.BatchNorm2d(8),
                nn.Conv2d(8, 16, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.AvgPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),
                nn.Conv2d(16, 16, (1, 16), (1, 1)),
                nn.BatchNorm2d(16), 
                nn.ELU(),
                # nn.AvgPool2d((1, 2), (1, 2)),
                nn.Dropout2d(0.5)
            )
        
class TSconv(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps)
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 40, (1, 25), (1, 1)),
                nn.AvgPool2d((1, 51), (1, 5)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.Conv2d(40, 40, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.Dropout(0.5),
            )
    