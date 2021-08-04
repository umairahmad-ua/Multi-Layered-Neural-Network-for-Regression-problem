from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split,GridSearchCV
import numpy as np
import pandas as pd
import torch.nn as nn
import torch as torch
from itertools import chain
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt

#Data_Reading
Xtrain=np.loadtxt("TrainData.csv") 
Ytrain=np.loadtxt("TrainLabels.csv")

    
    
    
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


# Get Root Mean Square Error
def get_RMSE(X_test,y_test, model):
    predictions =  list()
    test_dl = torch.utils.data.DataLoader(X_test,batch_size=100, shuffle=False)
    for i, (inputs) in enumerate(test_dl):
        yhat = model(inputs.float())
        yhat = yhat.detach().numpy()
        yhat = yhat.round()
        predictions.append(yhat)
    y_pred = list(chain.from_iterable(predictions))
    return (mean_squared_error(y_test.round(), y_pred,squared=False))



#Dataloader
def dataloader(X_train,y_train):
    train_data = []
    for i in range(len(X_train)):
        train_data.append([X_train[i], y_train[i]])
    return train_data



#Kfold and optimization 
# Optimization of hyper parameters
batch_size = 100
hyper_params={
    "learning_rate":[0.1]
}
kfold_cross_validation = 5
models_result =  list()
lr_result =  list()
kfold = KFold(n_splits=kfold_cross_validation,random_state=None, shuffle=False)




for train_index, test_index in kfold.split(Xtrain):
    X_train, X_test = Xtrain[train_index], Xtrain[test_index]
    y_train, y_test = Ytrain[train_index], Ytrain[test_index]
    train_data=dataloader(X_train,y_train)
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size, shuffle=True)
    net = Networks()
    net.train()  
    learning_rate_min, learning_rate_max = .005, .001
    no_parameters = 10
    hyper_params["learning_rate"] = np.linspace(learning_rate_min, learning_rate_max, no_parameters)
    
    
    for lrn_rate in hyper_params["learning_rate"]:
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
        print("\nDone with epochs")
        
        
        models_result.append(get_RMSE(X_test,y_test,net))
        lr_result.append(lrn_rate)
    print("next fold")
    
    
    
    
print(min(models_result))
i=0
for x in models_result:
    if(x==min(models_result)):
        print(str(i)+ "    "+str(lr_result[i])+"   " +str(x))
    i=i+1