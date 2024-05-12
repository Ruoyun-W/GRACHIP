import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn.models import GCN
from torch_geometric.nn.models import GAT

import random
import sys
import copy
from torch_geometric.data import HeteroData
from tqdm import tqdm
import json
from itertools import combinations
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DataParallel, GCNConv

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import torch.optim as optim
import math

from torch.cuda.amp import autocast, GradScaler
from scipy.stats import spearmanr, pearsonr
import copy

feature_num = 6
GCN_layer = 3
batchSize = 16
chunksize = 200        
samplesize = 200 
sample_stride = 50
chunk_stride = 100   
gcnInputsize = 768
gcnHiddensize = 400
gcnOutputsize = 768
edge_attr_scaler = 100
edge_portion = 0.20


def signal_handler(signal, frame):
    # Perform necessary cleanup here, such as closing files or connections.
    # Then exit the process.
    sys.exit(0)

def reproducible_computation():
    # Set seeds to ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # # Example computation that involves randomness
    # tensor = torch.rand(5)  # Random tensor generation
    # array = np.random.rand(5)  # Random numpy array generation
    # random_number = random.random()  # Random float from the random module

    # print("Torch tensor:", tensor)
    # print("Numpy array:", array)
    # print("Random float:", random_number)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)  # Output: 16 x 100 x 100
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1) # Output: 32 x 50 x 50
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) # Output: 64 x 25 x 25
        self.fc1 = nn.Linear(32*25*25,1024)
        self.fc2 = nn.Linear(1024,256)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.fc1(x.view(x.size(0),-1)))
        x = torch.relu(self.fc2(x))
        return x

class network(torch.nn.Module):
    def __init__(self,indexsize,gcnInputsize,gcnHiddensize,gcnOutputsize,batchSize):
        super(network, self).__init__()

        pe = torch.zeros(chunksize,768).float()
        position = torch.arange(0,chunksize).float().unsqueeze(1)
        div_term = (torch.arange(0,768,2).float()* - (math.log(10000.0)/768)).exp()
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        self.pe = pe.to(device)
        self.dense1 = nn.Linear(indexsize,768)
        self.dense2 = nn.Linear(768,768)

        self.conv1 = GCN(gcnInputsize, gcnHiddensize,GCN_layer, out_channels=gcnOutputsize)

        self.transform_conv1 = nn.Conv2d(in_channels=2*gcnOutputsize, out_channels=int(1.5*gcnOutputsize), kernel_size=3, stride=1,padding=1)
        self.transform_conv2 = nn.Conv2d(in_channels=int(1.5*gcnOutputsize), out_channels=gcnOutputsize//2, kernel_size=3, stride=1,padding=1)
        self.transform_conv3 = nn.Conv2d(in_channels=gcnOutputsize//2, out_channels=gcnOutputsize//8, kernel_size=3, stride=1,padding=1)
        self.transform_conv4 = nn.Conv2d(in_channels=gcnOutputsize//8, out_channels=gcnOutputsize//16, kernel_size=3, stride=1,padding=1)
        self.transform_conv5 = nn.Conv2d(in_channels=gcnOutputsize//16, out_channels=gcnOutputsize//32, kernel_size=3, stride=1,padding=1)
        self.transform_conv6 = nn.Conv2d(in_channels=gcnOutputsize//32, out_channels=gcnOutputsize//64, kernel_size=3, stride=1,padding=1)
        self.transform_conv7 = nn.Conv2d(in_channels=gcnOutputsize//64, out_channels=1, kernel_size=1, stride=1,padding=0)
        self.pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.linear1=torch.nn.Linear(2*gcnOutputsize,gcnOutputsize)
        self.linear2 = torch.nn.Linear(gcnOutputsize, 1)

        self.indexsize=indexsize       
        self.batchsize=batchSize
        self.bn = nn.LayerNorm([200,gcnOutputsize])  # BatchNorm1d layer

        self.transformer_layer = nn.TransformerEncoderLayer(d_model=gcnInputsize, nhead=12, batch_first=True)
        self.transformer_layers = [copy.deepcopy(self.transformer_layer).to(device) for i in range(12)]
        self.gcnInputsize=gcnInputsize

        self.encoder = Encoder()
        self.encoder.load_state_dict(torch.load("./encoder.pt"))
        self.encoder.eval()
        for i in self.encoder.parameters():
            i.requires_grad=False
        self.fc1 = nn.Linear(768+indexsize,768)

    def forward(self, data, device = "cuda" if torch.cuda.is_available() else "cpu", batchSize = batchSize,edge_portion = 0.2,eval = False):
        if eval:
            x1, x2, edge_index,edge_attr, lbl = data.x1, data.x2, data.edge_index, data.edge_attr, None
        else:
            x1, x2, edge_index,edge_attr, lbl = data.x1, data.x2, data.edge_index, data.edge_attr, torch.nan_to_num(data.y).float()
        x1 = x1.view(batchSize, samplesize, -1)
        x2 = x2.view(batchSize, samplesize, -1)

        x = torch.cat([x1,x2],2)
        x = self.fc1(x) + self.pe

        indices = np.random.choice(edge_index.size(1), size=int(edge_index.size(1)*edge_portion), replace=False)
        indices = torch.tensor(indices).to(device)
        edge_index_sl = torch.gather(edge_index, 1, torch.stack([indices,indices],0))

        edge_attr_sl = torch.gather(edge_attr, 0, indices)/edge_attr_scaler
        x = self.conv1(x.view(-1,768), edge_index_sl, edge_weight=edge_attr_sl)

        x = x.view(batchSize,samplesize,-1) + self.pe
        for layers in self.transformer_layers:
            x1 = layers(x)            
            x1 = self.bn(x1+x)

            x1 = F.gelu(x1)
            x = x1


        origs = x.unsqueeze(2).expand(x.size(0), x.size(1), x.size(1), -1).reshape(x.size(0), x.size(1) * x.size(1), -1)
        dests = x.unsqueeze(1).expand(x.size(0), x.size(1), x.size(1), -1).reshape(x.size(0), x.size(1) * x.size(1), -1)
        embs = torch.cat([origs, dests], dim=2)


        embs = embs.permute(0, 2, 1).contiguous()

        embs = embs.view(batchSize,-1,samplesize,samplesize)
        x = self.transform_conv1(embs)
        x = F.gelu(self.pool(x))
        x = self.transform_conv2(x)
        x = F.gelu(self.pool(x))
        x = self.transform_conv3(x)
        x = F.gelu(self.pool(x))
        x = self.transform_conv4(x)
        x = F.gelu(self.pool(x))
        x = self.transform_conv5(x)
        x = F.gelu(self.pool(x))
        x = self.transform_conv6(x)
        x = F.gelu(x)
        x = self.transform_conv7(x)


        x = x.permute(0, 2, 3, 1).contiguous()

        if eval:
            enc_pred = None
            enc_lbl = None
        else:
            enc_pred = self.encoder(x.view(batchSize,1,samplesize,samplesize))

            enc_lbl = self.encoder(lbl.view(batchSize,1,samplesize,samplesize))

        return x.view(-1), enc_pred, enc_lbl