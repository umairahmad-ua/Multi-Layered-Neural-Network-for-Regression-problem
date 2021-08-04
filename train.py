from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split,GridSearchCV
import numpy as np
import pandas as pd
import torch.nn as nn
import torch as torch
from itertools import chain
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt


Xtrain=np.loadtxt("TrainData.csv") 
Ytrain=np.loadtxt("TrainLabels.csv")
train_data = []
for i in range(len(Xtrain)):
    train_data.append([Xtrain[i], Ytrain[i]])


# Network Architechture
class Networks(nn.Module):
    def __init__(self):
        super(Networks,self).__init__()
        self.l1 = nn.Linear(8, 20)  
        self.l2 = nn.Linear(20, 20)
        self.l3 = nn.Linear(20, 20)
        self.l4 = nn.Linear(20, 20)
        self.l5 = nn.Linear(20, 20)
        self.ouput = nn.Linear(20, 1)
        
    def forward(self, x):
        z = torch.relu(self.l1(x))
        z = torch.relu(self.l2(z))
        z = torch.relu(self.l3(z))
        z = torch.relu(self.l4(z))
        z = torch.relu(self.l5(z))
        z = self.ouput(z)  # no activation
        return z
    
#traning
models_result =  list()
bat_size = 100
train_loader = torch.utils.data.DataLoader(train_data,batch_size=bat_size, shuffle=True)

net = Networks()
net.train()  
lrn_rate = 0.0045
loss_func = torch.nn.MSELoss()
#optimizer = torch.optim.SGD(net.parameters(), lr=lrn_rate, momentum=0.9)
optimizer = torch.optim.Adam(net.parameters(),lr=lrn_rate)

for epoch in range(0, 500):
    for data in train_loader:
        X = data[0]  
        Y = data[1] 
        optimizer.zero_grad()
        oupt = net(X.float())
        Y=Y.view(Y.shape[0],1)
        loss_val = loss_func(oupt, Y.float()) 
        loss_val.backward()
        optimizer.step()
torch.save(net.state_dict(), "my Model")