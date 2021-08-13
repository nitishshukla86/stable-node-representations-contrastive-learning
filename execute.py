import argparse
import torch
from utils import *
#from models import SNR 
from torch.geometric.nn import GCNConv

import torch.nn as nn
from torch_geometric.utils import dropout_adj, convert
parser = argparse.ArgumentParser("SNR")

parser.add_argument('--dataset',          type=str,           default="",                help='data')
parser.add_argument('--aug_type',         type=str,           default="",                help='aug type: mask or perturb')
parser.add_argument('--drop_percent',     type=float,         default=0.1,               help='drop percent')
parser.add_argument('--seed',             type=int,           default=39,                help='seed')
#parser.add_argument('--gpu',              type=int,           default=0,                 help='gpu')
parser.add_argument('--save_name',        type=str,           default='try.pkl',                help='save ckpt name')
parser.add_argument('--num_hidden',             type=int,           default=32,                help='number of hidden layers in classifier')
parser.add_argument('--num_epochs',             type=int,           default=64,                help='number of epochs')


args = parser.parse_args()
print('-' * 100)
print(args)
print('-' * 100)

dataset = args.dataset
aug_type = args.aug_type
drop_percent = args.drop_percent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.dataset == 'german':
    dataset = 'german'
    sens_attr = "Gender"
    predict_attr = "GoodCustomer"
    label_number = 100
    sens_number = 100
    path = "./dataset/german"
    test_idx = True        
    adj, features, labels, idx_train, idx_val, idx_test,sens, idx_sens_train = load_german(dataset,
                                                                                sens_attr,
                                                                                predict_attr,
                                                                                path=path,
                                                                                label_number=label_number,
                                                                                sens_number=sens_number)

class SNR(nn.Module):
    def __init__(self,nfeat,nhid,nclass):
        super(SNR,self).__init__()
        self.gcn=GCNConv(nfeat,nhid)
        self.fc=nn.Linear(nhid,nclass)


    def forward(self, x, edge_index):
        x = self.gcn(x, edge_index)
        x = self.fc(x)
        return x

model=SNR(nfeat=features.shape[0],nhid=32,nclass=1)
edge_index = convert.from_scipy_sparse_matrix(adj)[0]
criterion=nn.BCEWithLogitsLoss()
optimizer=torch.optim.Adam(model.parameeters(),lr=0.1,weight_decay=0.01)

n_epoch=args.n_epochs

for epoch in range(n_epoch):
    model.train()
    pred=model(features,edge_index)
    loss=criterion(pred,labels)
    optimizer.step()
    optimizer.zero_grad()









