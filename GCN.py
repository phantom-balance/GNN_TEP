from construct_graph import graph_constructor
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import DenseGCNConv as GCNConv
# from torch_geometric.loader import DataLoader
from torch_geometric.loader import DenseDataLoader
from loader import TEP 
import einops
import pickle
import os

from utils import create_data, save_checkpoint, check_accuracy, load_checkpoint


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_nodes = 82 
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
directed = False
reverse_directed = False

load_model = False
save_model = True

# datas, lengths = graph_constructor(IDVs=IDVs, sequence_length=sequence_length, modes="Mode1", intensities=intensities, runss=runs)


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(sequence_length, embedding_size)
        self.relu1 = nn.ReLU()
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(embedding_size*num_nodes, 300)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.out = nn.Linear(300, num_classes)

    def forward(self, x, adj):
        updated_node = self.conv1(x, adj)
        updated_node = self.relu1(updated_node)
        updated_node = self.conv2(updated_node, adj)
        updated_node = self.relu2(updated_node)
        latent_feature = einops.rearrange(updated_node, 'b n x -> b (n x)')
        latent_feature = self.fc1(latent_feature)
        latent_feature = self.relu3(latent_feature)
        latent_feature = self.dropout(latent_feature)
        out = self.out(latent_feature)

        return out

model = GCN().to(device)
print(model)
print("Number of parameters:", sum(p.numel() for p in model.parameters()))

# x = torch.rand(batch_size, num_nodes, sequence_length)
# model.eval()
# adj_temp = einops.repeat(adj, 'r c -> b r c', b=batch_size)
# print(model(x, adj_temp).shape)
# model.train()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


if (directed==False and reverse_directed==False): # Undirected
    adj_ = pickle.load(open("processed_data/undirected_adjacency_matrix.p", "rb"))
    adj_ = torch.from_numpy(adj_).float()
    filename = f"processed_data/GCN_TEP_{sequence_length}_undirected.pth.tar"
    if load_model:
        load_checkpoint(torch.load(filename, map_location=device), model, optimizer)#, map_location=device)


elif (directed==True and reverse_directed==False):
    adj_ = pickle.load(open("processed_data/directed_adjacency_matrix.p", "rb"))
    adj_ = torch.from_numpy(adj_).float()
    filename = f"processed_data/GCN_TEP_{sequence_length}_directed.pth.tar"

elif (directed==False and reverse_directed==True):
    adj_ = pickle.load(open("processed_data/directed_adjacency_matrix.p", "rb"))
    adj_ = torch.from_numpy(adj_).float()
    filename = f"processed_data/GCN_TEP_{sequence_length}_reversedirected.pth.tar"
    
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
    # train(num_epochs, train_loader, adj = adj_)