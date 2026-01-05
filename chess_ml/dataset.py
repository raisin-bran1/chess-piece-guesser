# Pytorch Dataset class
from torch.utils.data import Dataset
import torch

class ChessDataset(Dataset):
    def __init__(self, data_path, n_samples=None, eval = False):
        data = torch.load(data_path, weights_only=False)
        if n_samples is None:
            self.X = data["inputs"]
            self.Y = data["targets"]
            self.moves = data["moves"]
        else:
            self.X = data["inputs"][:n_samples]
            self.Y = data["targets"][:n_samples]
            self.moves = data["moves"][:n_samples]
        self.max_moves = torch.max(self.moves)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        if eval:
            return self.X[idx], self.Y[idx], self.moves[idx]
        else:
            return self.X[idx], self.Y[idx]