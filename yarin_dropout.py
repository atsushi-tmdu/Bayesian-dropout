#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import pandas as pd
import torch.optim.lr_scheduler
from model import BAReg
torch.cuda.is_available()

#%%
data_early = pd.read_pickle("/home/work/work/Bayesian_yarin/data/data_early.pkl")
data_later = pd.read_pickle("/home/work/work/Bayesian_yarin/data/data_later.pkl")

data_early=torch.from_numpy(data_early.values)
data_later=torch.from_numpy(data_later.values)



#%%
input_dim = 1
output_dim =1
hidden_dim = 1024

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = BAReg(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(),lr = 0.01, momentum =0.9, weight_decay=1e-6)

gamma = 0.00001
p = 0.25

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: (1+gamma*epoch)**-p)


nn.init.uniform_(net.fc1.weight,-np.sqrt(3.0/input_dim),np.sqrt(3.0/input_dim))
nn.init.uniform_(net.fc2.weight,-np.sqrt(3.0/hidden_dim),np.sqrt(3.0/hidden_dim))
nn.init.uniform_(net.fc3.weight,-np.sqrt(3.0/hidden_dim),np.sqrt(3.0/hidden_dim))
nn.init.uniform_(net.fc4.weight,-np.sqrt(3.0/hidden_dim),np.sqrt(3.0/hidden_dim))

nn.init.constant_(net.fc1.bias, 0.0)
nn.init.constant_(net.fc2.bias, 0.0)
nn.init.constant_(net.fc3.bias, 0.0)
nn.init.constant_(net.fc4.bias, 0.0)


num_epochs = 1000000

path = '/home/work/work/Bayesian_yarin/result/loss2.txt'
log_path = 'home/work/work/Bayesian_yarin/result/log.txt'
model_path='home/work/work/Bayesian_yarin/model/log.txt'
#%%
for epoch in range(num_epochs):
    
    train_loss = 0
     
    #train===================
    net.train()
    
    x = data_early[:,0].to(device).float()
    y = data_early[:,1].to(device).float()
    optimizer.zero_grad()
        
    x = x.unsqueeze(1)
    y_pred = net(x).squeeze()
    loss = criterion(y_pred,y)
    train_loss += loss.item()
    loss.backward()
    optimizer.step()
        
       
        
    
    scheduler.step()

    if (epoch %10000 ==0):
        with open(path, mode='a') as f:
            if (epoch ==0):
                f.write('epoch,train_loss')
                
            f.write('\n{},{}'.format(epoch,train_loss))
  

    if (epoch % 10000==0):
        print("epoch:{},loss:{}".format(epoch,loss))


file = "model_early.prf"
torch.save(net.state_dict(),file, pickle_protocol=4 )
