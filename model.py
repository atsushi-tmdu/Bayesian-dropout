import torch.nn as nn
import torch.nn.functional as F

class BAReg (nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
            super(BAReg, self).__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.hidden_dim = hidden_dim
            self.fc1 = nn.Linear(input_dim, hidden_dim,bias=True)
            self.fc2 = nn.Linear(hidden_dim,hidden_dim,bias=True)
            self.fc3 = nn.Linear(hidden_dim,hidden_dim,bias=True)
            self.fc4 = nn.Linear(hidden_dim,output_dim,bias=True)
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