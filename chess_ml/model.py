import torch.nn as nn

# Standard pytorch MLP

class MLP(nn.Module):
    def __init__(self, layers: list[int]):
        super().__init__()
        self.model = nn.Sequential()
        for i in range(1, len(layers)):
            self.model.add_module(f"dense{i}", nn.Linear(layers[i-1], layers[i]))
            if i != len(layers) - 1:
                self.model.add_module(f"act{i}", nn.ReLU())

    def forward(self, x): # turns (B, 64 * 13) into (B, 64, 13)
        X = self.model(x) 
        if X.ndim == 1:
            return X.view(64, 13)
        return X.view(X.size(0), 64, 13)


# My own transformer implementation

# Takes in a tensor (B, tokens, embedding_dim)
# Obtains K, Q, V each as (B, tokens, [new dim]) via linear layers (automatically only changes last dimension)
# return softmax(Q * K_T / sqrt(k_dim)) * V
# note that * and _T should act as standard matrix mult & transpose on the last two dimensions of the tensor

class Attention(nn.Module):
    def __init__(self):
        super().__init__()

# Reshape the tensor from (B, tokens, embed = heads * k) to (B, heads, tokens, k)
# Apply Attention.forward
# Undo the reshape and return the tensor

class MultiHeadAttention(Attention):
    def __init__(self):
        super().__init__()

# One transformer block, consisting of a MH-attention block and a MLP block (with same input & output dim)

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

# Embeds input, runs a # of identical transformer blocks, final linear output layer

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()


# Baking in some explicit parameters

class MLP_basic(MLP):
    def __init__(self):
        super().__init__([192, 512, 256, 128, 64 * 13])

class MLP_big(MLP):
    def __init__(self):
        super().__init__([192, 1536, 1024, 512, 64 * 13])