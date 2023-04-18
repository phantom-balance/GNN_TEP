import os
from torch.utils.data import random_split
from construct_graph import graph_constructor
from loader import TEP
import torch
import einops


def create_data(Mode, intensities, IDVs, run, sequence_length, seed:int=42):
    if os.path.isfile(f"processed_data/Run{run}_Sequence_length_{sequence_length}_seed{seed}.pt"):
        print(f"graph data exists for sequence_length={sequence_length} run={run} seed={seed}")
        graph_data = torch.load(f"processed_data/Run{run}_Sequence_length_{sequence_length}_seed{seed}.pt")
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)

        print(f"Creating graph data for sequence_length={sequence_length} run={run}")
        data_list, length = graph_constructor(IDVs=IDVs, sequence_length=sequence_length, modes=Mode, intensities=intensities, runs=[run])
        data = TEP(graph_data=data_list, length=length)
        train_size = int(0.8*length)
        test_size = length - train_size
        train_data, test_data = random_split(dataset=data, lengths=[train_size, test_size], generator=generator)

        graph_data = {'train': train_data,
                      'test': test_data}
        torch.save(graph_data, f"processed_data/Run{run}_Sequence_length_{sequence_length}_seed{seed}.pt")

def save_checkpoint(state, filename):
    print("__Saving Checkpoint__")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("__Loading Checkpoint__")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def check_accuracy(loaded_data, model, adj_, device):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for batch in loaded_data:
            shape_dict = einops.parse_shape(batch.y, 'b') 
            batch = batch.to(device)
            adj = einops.repeat(adj_, 'r c -> b r c', b=shape_dict['b'])
            scores = model(batch.x.float(), adj)
            _, prediction = scores.max(1)
            num_correct += (prediction==batch.y).sum()
            num_samples += prediction.size(0)
        # print(f'Got {num_correct}/{num_samples} correct, prediction rate={float(num_correct)/float(num_samples)*100:.3f}')
    model.train()

    accuracy = num_correct/num_samples

    return accuracy, num_samples, num_correct


def summary_return():
    pass
