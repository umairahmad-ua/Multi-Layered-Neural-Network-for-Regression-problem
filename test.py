import numpy as np
import pandas as pd
import torch.nn as nn
import torch as torch
from itertools import chain


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



def get_predicitions(X_test, model):
    predictions =  list()
    test_dl = torch.utils.data.DataLoader(X_test,batch_size=100, shuffle=False)
    for i, (inputs) in enumerate(test_dl):
        yhat = model(inputs.float())
        yhat = yhat.detach().numpy()
        yhat = yhat.round()
        predictions.append(yhat)
    y_pred = list(chain.from_iterable(predictions))
    return y_pred
    
    


Xtest=np.loadtxt("TestData.csv")
model = Networks()
model.load_state_dict(torch.load("my Model"))
model.eval()
predictions=get_predicitions(Xtest,model)
np.savetxt("i212081_Predictions.csv", predictions)
pre=np.loadtxt("i212081_Predictions.csv")
print(pre)