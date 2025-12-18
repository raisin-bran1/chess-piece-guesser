# Pytorch Dataset class
from torch.utils.data import Dataset
import torch

class ChessDataset(Dataset):
    def __init__(self, data_path, n_samples=None):
        data = torch.load(data_path, weights_only=False)
        if n_samples is None:
            self.X = data["inputs"]
            self.Y = data["targets"]
        else:
            self.X = data["inputs"][:n_samples]
            self.Y = data["targets"][:n_samples]

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]