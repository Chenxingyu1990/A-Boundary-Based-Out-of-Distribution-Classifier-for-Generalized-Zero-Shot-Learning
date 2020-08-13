import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import ipdb

def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


class Attr_Encoder(nn.Module):
    def __init__(self, input_size, mid_size, hidden_size):
        super(Attr_Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, mid_size, bias = True) 
        self.relu1 = nn.ReLU()      
        self.fc2 = nn.Linear(mid_size, hidden_size, bias = True)
        self.fc3 = nn.Linear(mid_size, 1, bias = False)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x2 = self.fc2(x)
        x2 = x2 / x2.norm(dim=-1, keepdim=True)
        x3 = self.fc3(x) 
        x3 = F.softplus(x3) + 100.0
        return x2, x3
        
class Attr_Decoder(nn.Module):
    def __init__(self, hidden_size, mid_size, output_size):
        super(Attr_Decoder, self).__init__()
        self.fc1 = nn.Linear(hidden_size, mid_size, bias = True)       
        self.fc2 = nn.Linear(mid_size, output_size, bias = True) 
        self.relu = nn.ReLU()
        
  
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        
      
        return x

class Encoder(nn.Module):
    def __init__(self, input_size, mid_size, hidden_size):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, mid_size, bias = True)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(mid_size , hidden_size, bias = True)
        self.fc3 = nn.Linear(mid_size , 1, bias = False)
       
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu2(x)
        x2 = self.fc2(x)
        x2 = x2 / x2.norm(dim=-1, keepdim=True)
        x3 = self.fc3(x)
        x3 = F.softplus(x3) + 100.0
        return x2, x3
        

class Decoder(nn.Module):
    def __init__(self, hidden_size, mid_size, output_size):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(hidden_size, mid_size, bias = True)
        self.fc2 = nn.Linear(mid_size, output_size, bias = True)
        self.relu1 = nn.ReLU()
    def forward(self, x):
    
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        
        return x
        
class Decoder_Imagenet(nn.Module):
    def __init__(self, hidden_size, mid_size, output_size):
        super(Decoder_Imagenet, self).__init__()

        self.fc1 = nn.Linear(hidden_size, mid_size, bias = True)
        self.fc2 = nn.Linear(mid_size, output_size, bias = True)
        self.relu1 = nn.LeakyReLU(0.2)
    def forward(self, x):
    
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        
        return x
        
class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
    def forward(self, x): 
        o = self.logic(self.fc(x))
        return o         
