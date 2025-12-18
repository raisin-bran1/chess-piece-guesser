import torch.nn as nn

class MLP(nn.module):
    def __init__(self, layers: list[int]):
        super().__init__()
        self.model = nn.Sequential()
        for i in range(1, len(layers)):
            self.model.add_module(f"dense{i}", nn.Linear(layers[i-1], layers[i]))
            if i != len(layers) - 1:
                self.model.add_module(f"act{i}", nn.ReLU())

    def forward(self, x):
        return self.model(x)