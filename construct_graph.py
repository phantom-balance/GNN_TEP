import einops
import torch
import numpy as np
from torch_geometric.data import Data
from get_data import get_data

def graph_constructor(IDVs, sequence_length, modes="Mode1", intensities=None, runs=None):
    print("STARTING_constructing graphs")
    data_list = get_data(mode=modes, IDV=IDVs, intensities=intensities, runs=runs)

    length = 0
    X_all = []
    Y_all = []
    for i, data in enumerate(data_list):
        X_temp = data[:, :-1]
        Y_temp = data[:, -1]
        if i == 0:
            Y_all = Y_temp.copy()
            Y_true = Y_all[sequence_length-1:]

            X_all = X_temp.copy()
            X_all = torch.from_numpy(X_all)
            X_true = X_all.unfold(0, sequence_length, 1).unfold(1, 52, 1)
            X_tensor = einops.rearrange(X_true, 'a b c d -> (a b) c d')

        else:
            Y_all = Y_temp.copy()
            Y_ = Y_all[sequence_length-1:]
            Y_true = np.concatenate((Y_true, Y_), axis=0)

            X_all = X_temp.copy()
            X_all = torch.from_numpy(X_all)
            X_ = X_all.unfold(0, sequence_length, 1).unfold(1, 52, 1)
            X_ = einops.rearrange(X_, 'a b c d -> (a b) c d')
            X_tensor = torch.cat((X_tensor, X_), axis=0)
        
        length += data.shape[0] - sequence_length + 1

    Y_tensor = torch.from_numpy(Y_true)
    graph_data_list = []

    for i in range(length):
        node_initialization = torch.zeros(sequence_length, 30)
        X_tensor_temp = X_tensor[i]
        X_new = torch.cat((X_tensor_temp, node_initialization), axis=1)
        X_new = einops.rearrange(X_new, 'f n -> n f')
        data_instance = Data(x = X_new, y = Y_tensor[i])
        graph_data_list.append(data_instance)
    
    print("ENDING_constructing graphs")
    
    return graph_data_list, length    