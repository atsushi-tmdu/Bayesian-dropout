#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import pandas as pd
import torch.optim.lr_scheduler
torch.cuda.is_available()
from model import BAReg


#%%
data_early = pd.read_pickle("data/data_early.pkl")
data_later = pd.read_pickle("data/data_later.pkl")

data_early=torch.from_numpy(data_early.values)
data_later=torch.from_numpy(data_later.values)

#%%
input_dim = 1
output_dim =1
hidden_dim = 1024

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = BAReg(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)

#%%
file = "model_early.prf"
net.load_state_dict(torch.load(file))


#%%
x= data_early[:,0].to(device).float()
y= data_early[:,1].to(device).float()

#%%
x = x.unsqueeze(1)

#%%
y_pred = net(x).squeeze().detach().cpu().numpy()

#%%
y_pred
#%%
import matplotlib.pyplot as plt
plt.plot(x.squeeze().cpu().numpy(),y_pred)

#%%
plt.plot(x.squeeze().cpu().numpy(),y.cpu().numpy())

#%%
plt.plot(x.squeeze().cpu().numpy(),y_pred)
plt.plot(x.squeeze().cpu().numpy(),y.cpu().numpy())

#%%
len(x)

#%%
result_epochs = 10

y_pred_array=np.zeros((result_epochs,len(x)))

for epoch in range(result_epochs):

    net.train()

    x = data_early[:,0].to(device).float()
    y = data_early[:,1].to(device).float()

    x = x.unsqueeze(1)
    y_pred = net(x).squeeze().detach().cpu().numpy()
    y_pred_array[epoch] = y_pred



#%%
y_pred_array

#%%
