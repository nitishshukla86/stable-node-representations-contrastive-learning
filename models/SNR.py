import torch.nn as nn
from torch.geometric.nn import GCNConv


class SNR(nn.Module):
    def __init__(self,nfeat,nhid,nclass):
        super(self,SNR).__init__()
        self.gcn=GCNConv(nfeat,nhid)
        self.fc=nn.Linear(nhid,nclass)


    def forward(self, x, edge_index):
        x = self.gcn(x, edge_index)
        x = self.fc(x)
        return x
