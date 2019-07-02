#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
torch.cuda.is_available()

#%%
data_early = pd.read_pickle("/home/work/work/Bayesian_yarin/data/data_early.pkl")
data_later = pd.read_pickle("/home/work/work/Bayesian_yarin/data/data_later.pkl")

data_early=torch.from_numpy(data_early.values)
data_later=torch.from_numpy(data_later.values)

train_loader = torch.utils.data.DataLoader(dataset=data_early, batch_size=1, shuffle=True,num_workers=4)
test_loader = torch.utils.data.DataLoader(dataset=data_later, batch_size=1, shuffle=False,num_workers=4)

#%%
class BAReg (nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
            super(BAReg, self).__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.hidden_dim = hidden_dim
            self.fc1 = nn.Linear(input_dim, hidden_dim,bias=False)
            self.fc2 = nn.Linear(hidden_dim,hidden_dim,bias=False)
            self.fc3 = nn.Linear(hidden_dim,hidden_dim,bias=False)
            self.fc4 = nn.Linear(hidden_dim,output_dim,bias=False)
            self.dropout1 = nn.Dropout(0.1)
            self.dropout2 = nn.Dropout(0.1)
            self.dropout3 = nn.Dropout(0.1)
    
    def forward(self,x):
            x = F.relu(self.fc1(x))
            x = self.dropout1(x)
            x = F.relu(self.fc2(x))
            x = self.dropout2(x)
            x = F.relu(self.fc3(x))
            x = self.dropout3(x)
            x = self.fc4(x)
            return x

#%%
input_dim = 1
output_dim =1
hidden_dim = 1024

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = BAReg(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(),lr = 0.00001, momentum =0.9, weight_decay=1e-6)

nn.init.kaiming_normal_(net.fc1.weight)
nn.init.kaiming_normal_(net.fc2.weight)
nn.init.kaiming_normal_(net.fc3.weight)
nn.init.kaiming_normal_(net.fc4.weight)


num_epochs = 10000
result_epochs = 1000

path = '/home/work/work/Bayesian_yarin/result/loss2.txt'
#%%
for epoch in range(num_epochs):
    
    train_loss = 0
     
    #train===================
    net.train()
    
    for i, data in enumerate(train_loader):
        
        x = data[:,0].to(device).float()
        y = data[:,1].to(device).float()
        optimizer.zero_grad()
        
       
        y_pred = net(x)
        loss = criterion(y_pred,y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        #print(net(x))
        #print(("y_pred:{}, y:{}".format(y_pred.cpu(), y.cpu())))
        #print("loss {}".format(loss.item()))
        #print("i:{}  train_loss:{}".format(i,train_loss))
        #print("-------------")
        
    if (epoch %400 ==0):
        avg_train_loss = train_loss/len(train_loader.dataset)
        with open(path, mode='a') as f:
            if (epoch ==0):
                f.write('epoch,train_loss')
                
            f.write('\n{},{}'.format(epoch,train_loss))
        params = net.state_dict()
        file = "model_epoch_{}.prm".format(epoch)
        torch.save(params,file,pickle_protocol=4)   


#%%
