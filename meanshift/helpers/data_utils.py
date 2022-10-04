import torch

def to_dataloader(X,y, bs=20):
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=bs, shuffle=True)