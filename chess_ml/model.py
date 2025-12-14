import torch
import torch.nn as nn
import numpy as np

class MLP(nn.module):
    def __init__(self, layers: list[int]):
        self.model = nn.Sequential()
        for i in range(1, len(layers)):
            self.model.add_module(f"dense{i}", nn.Linear(layers[i-1], layers[i]))
            if i != len(layers) - 1:
                self.model.add_module(f"act{i}", nn.ReLU())