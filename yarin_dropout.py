#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import pandas as pd
import torch.optim.lr_scheduler
#from matplotlib import pyplot as plt
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
optimizer = optim.SGD(net.parameters(),lr = 0.0001, momentum =0.9, weight_decay=1e-6)

gamma = 0.00001
p = 0.25

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: (1+gamma*epoch)**-p)


nn.init.uniform_(net.fc1.weight,-np.sqrt(3.0/input_dim),np.sqrt(3.0/input_dim))
nn.init.uniform_(net.fc2.weight,-np.sqrt(3.0/hidden_dim),np.sqrt(3.0/hidden_dim))
nn.init.uniform_(net.fc3.weight,-np.sqrt(3.0/hidden_dim),np.sqrt(3.0/hidden_dim))
nn.init.uniform_(net.fc4.weight,-np.sqrt(3.0/hidden_dim),np.sqrt(3.0/hidden_dim))


num_epochs = 41
result_epochs = 10000

path = '/home/work/work/Bayesian_yarin/result/loss2.txt'
log_path = 'home/work/work/Bayesian_yarin/result/log.txt'
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
        
        with open (log_path, mode='a') as f:
            f.write("{}\n".format(net(x).cpu().detach().numpy()))
            f.write("y_pred:{}, y:{}\n".format(y_pred.cpu(), y.cpu()))
            f.write("loss {}\n".format(loss.item()))
            f.write("i:{}  train_loss:{}\n".format(i,train_loss))
            f.write("-------------\n")
        
    
    scheduler.step()

    if (epoch %20 ==0):
        avg_train_loss = train_loss/len(train_loader.dataset)
        with open(path, mode='a') as f:
            if (epoch ==0):
                f.write('epoch,train_loss')
                
            f.write('\n{},{}'.format(epoch,train_loss))
        params = net.state_dict()
        file = "model_epoch_{}.prm".format(epoch)
        torch.save(params,file,pickle_protocol=4)   

    