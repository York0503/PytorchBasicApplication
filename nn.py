import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
 
# select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = "cpu" # run faster than cuda in some cases
print("Using {} device".format(device))
 
# Create a neural network
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(64*64, 512), # image length 64x64=4096,  fully connected layer
            nn.ReLU(), # try to take ReLU out to see what happen
            # nn.Linear(512, 512), # second hidden layer
            # nn.ReLU(),
            nn.Linear(512, 40) # 40 classes,  fully connected layer
            # nn.Softmax()
        )
    # Specify how data will pass through this model
    def forward(self, x):
        # out = self.mlp(x) 
 
        # Apply softmax to x here~
        x = self.mlp(x)
        out = F.log_softmax(x, dim=1) # it’s faster and has better numerical propertie than softmax
        # out = F.softmax(x, dim=1)
        return out
 
 
# define model, optimizer, loss function
model = MLP().to(device) # start an instance
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # default lreaning rate=1e-3
loss_fun = nn.CrossEntropyLoss() # define loss function
 
print(model)
input = torch.randn(100, 64 * 64) 
m1 = nn.Linear(64*64, 512)
output = m1(input)
print(output.size())
output = F.relu(output)
print(output.size())
m2 = nn.Linear(512, 40)
output = m2(output)
print(output.size())
output = F.log_softmax(output, dim=1)
print(output.size())