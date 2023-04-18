from construct_graph import graph_constructor
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import DenseGCNConv, dense_diff_pool
# from torch_geometric.loader import DataLoader
from torch_geometric.loader import DenseDataLoader
from loader import TEP 
import einops
import pickle
import os
from utils import create_data, save_checkpoint, check_accuracy, load_checkpoint
from math import ceil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_nodes = num_nodes = 82 
IDVs = np.arange(1, 29)
num_classes = 29

Mode = "Mode1"
intensities = [25, 50, 75, 100]
runs = np.arange(1, 101)

sequence_length = 10
learning_rate = 0.003
num_epochs = 1
batch_size = 2
embedding_size = 15
node_red_ratio = 0.6
directed = False
reverse_directed = False

load_model = True
save_model = True

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, lin=True):
        super().__init__()

        self.conv1 = DenseGCNConv(in_channels, hidden_channels)
        self.conv2 = DenseGCNConv(hidden_channels, hidden_channels)
        self.conv3 = DenseGCNConv(hidden_channels, out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2*hidden_channels+out_channels, out_channels)
        else:
            self.lin = None

    def forward(self, x, adj):
        batch_size, num_nodes, in_channels = x.size()
        x0 = x
        x1 = self.conv1(x0, adj).relu()
        x2 = self.conv2(x1, adj).relu()
        x3 = self.conv3(x2, adj).relu()

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = self.lin(x).relu()
        
        return x

class DiffPool(torch.nn.Module):
    def __init__(self):
        super().__init__()

        num_nodes = ceil(node_red_ratio*max_nodes)
        self.gnn1_pool = GNN(sequence_length, 15, num_nodes)
        self.gnn1_embed = GNN(sequence_length, 15, 15, lin=False)

        num_nodes = ceil(node_red_ratio*num_nodes)
        self.gnn2_pool = GNN(3*15, 15, num_nodes)
        self.gnn2_embed = GNN(3*15, 15, 15, lin=False)

        self.gnn3_embed = GNN(3*15, 15, 15, lin=False)

        self.lin1 = torch.nn.Linear(3*15, 30)
        self.out = torch.nn.Linear(30, num_classes)

    def forward(self, x, adj):
        s = self.gnn1_pool(x, adj)
        x = self.gnn1_embed(x, adj)

        x, adj, _, _ = dense_diff_pool(x, adj, s)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, _, _ = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.out(x)

        return x

model = DiffPool().to(device)
print(model)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

# x = torch.rand(batch_size, num_nodes, sequence_length)
# model.eval()
# adj_temp = einops.repeat(adj_, 'r c -> b r c', b=batch_size)
# print(model(x, adj_temp).shape)
# model.train()

if (directed==False and reverse_directed==False): # Undirected
    adj_ = pickle.load(open("processed_data/undirected_adjacency_matrix.p", "rb"))
    adj_ = torch.from_numpy(adj_).float()
    filename = f"processed_data/GCNdiffpool_TEP_{sequence_length}_undirected.pth.tar"
    if load_model:
        load_checkpoint(torch.load(filename, map_location=device), model, optimizer)#, map_location=device)


elif (directed==True and reverse_directed==False):
    adj_ = pickle.load(open("processed_data/directed_adjacency_matrix.p", "rb"))
    adj_ = torch.from_numpy(adj_).float()
    filename = f"processed_data/GCNdiffpool_TEP_{sequence_length}_directed.pth.tar"

elif (directed==False and reverse_directed==True):
    adj_ = pickle.load(open("processed_data/directed_adjacency_matrix.p", "rb"))
    adj_ = torch.from_numpy(adj_).float()
    filename = f"processed_data/GCNdiffpool_TEP_{sequence_length}_reversedirected.pth.tar"
    
else:
    print("check the boolean values of directed, undirected, and specify your scenario")


def train(num_epochs, train_loader, adj, save_every_batch_num:int=10, iterate:bool=False):
    if iterate:
        for epoch in num_epochs:
            batch = next(iter(train_loader))
            batch = batch.to(device)
            shape_dict = einops.parse_shape(batch, 'b n x') 
            adj = einops.repeat(adj_, 'r c -> b r c', b=shape_dict['b'])
            scores = model(batch.x.float(), adj)
            loss = criterion(scores, batch.y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(f'Epoch:{epoch} || Batch_idx:{batch_idx}')
            if save_model==True:
                if batch_idx % save_every_batch_num == 0:
                    checkpoint = {'state_dict': model.state_dict(),
                                  'optimizer': optimizer.state_dict()}
                    save_checkpoint(checkpoint, filename)

    else:
        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(train_loader):
                shape_dict = einops.parse_shape(batch.y, 'b') 
                
                batch = batch.to(device)
                adj = einops.repeat(adj_, 'r c -> b r c', b=shape_dict['b'])
                scores = model(batch.x.float(), adj)
                loss = criterion(scores, batch.y.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # print(f'Epoch:{epoch} || Batch_idx:{batch_idx}')
                if save_model==True:
                    if batch_idx % save_every_batch_num == 0:
                        checkpoint = {'state_dict': model.state_dict(),
                                    'optimizer': optimizer.state_dict()}
                        save_checkpoint(checkpoint, filename)

# IDVs = [1,2]
intensities = [25]
runs=[1]
for run in runs:
    print(f"RUN={run}")
    if os.path.isfile(f"processed_data/Run{run}_Sequence_length_{sequence_length}_seed42.pt"):
        graph_data = torch.load(f"processed_data/Run{run}_Sequence_length_{sequence_length}_seed42.pt")
    else:
        create_data(Mode, intensities, IDVs, run, sequence_length, seed=42)
        graph_data = torch.load(f"processed_data/Run{run}_Sequence_length_{sequence_length}_seed42.pt")
    
    # train_loader = DenseDataLoader(dataset=graph_data['train'], batch_size=batch_size, shuffle=True)
    test_loader = DenseDataLoader(dataset=graph_data['test'], batch_size=batch_size, shuffle=True)
    ac, tot, cor = check_accuracy(test_loader, model, adj_, device)
    print(f"ACCURACY={ac}, TOTAL={tot}, CORRECT={cor}")
    train(num_epochs, test_loader, adj = adj_)
    ac, tot, cor = check_accuracy(test_loader, model, adj_, device)
    print(f"ACCURACY={ac}, TOTAL={tot}, CORRECT={cor}")