import torch.nn as nn
import torch
import math

# Standard pytorch MLP

class MLP(nn.Module):
    def __init__(self, layers: list[int]):
        super().__init__()
        self.model = nn.Sequential()
        for i in range(1, len(layers)):
            self.model.add_module(f"dense{i}", nn.Linear(layers[i-1], layers[i]))
            if i != len(layers) - 1:
                self.model.add_module(f"act{i}", nn.ReLU())

    def forward(self, x):
        return self.model(x) 


# My own transformer implementation

# Takes in a tensor (B, tokens, embedding_dim)
# Obtains K, Q, V each as (B, tokens, embedding_dim) via linear layers (only acts on last dimension)
# return softmax(Q * K_T / sqrt(k_dim)) * V
# note that * and _T should act as standard matrix mult & transpose on the last two dimensions of the tensor

class Attention(nn.Module):
    def __init__(self, embed): # embed = query = key = value
        super().__init__()
        self.embed = embed
        self.wq, self.wk, self.wv = nn.Linear(embed, embed), nn.Linear(embed, embed), nn.Linear(embed, embed)
    
    def attention(self, Q, K, V):
        scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
        return torch.softmax(scores, dim=-1) @ V

    def forward(self, x):
        Q, K, V = self.wq(x), self.wk(x), self.wv(x)
        return self.attention(Q, K, V)

# Reshape from (B, tokens, embed = heads * d) to (B, heads, tokens, d)
# Apply attention
# Undo the reshape, return the tensor after one more linear layer

class MultiHeadAttention(Attention):
    def __init__(self, embed, heads):
        assert embed % heads == 0
        super().__init__(embed)
        self.heads = heads
        self.d = self.embed // self.heads
        self.out = nn.Linear(embed, embed)

    def reshape(self, x):
        return x.view(x.size(0), x.size(1), self.heads, self.d).transpose(1, 2)

    def forward(self, x): # x = (B, tokens, embed)
        Q, K, V = self.reshape(self.wq(x)), self.reshape(self.wk(x)), self.reshape(self.wv(x))
        att = self.attention(Q, K, V) # (B, heads, tokens, d)
        att_concat = att.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.embed) # (B, tokens, embed)
        return self.out(att_concat)

# One transformer block, consisting of a MH-attention block and a MLP block (with same input & output dim)
# Remember residual connection & layernorm! Instead of X -> F(X) do X -> X + F(LayerNorm(X))

class TransformerBlock(nn.Module):
    def __init__(self, embed, heads, mlp_layers):
        super().__init__()
        self.attention = MultiHeadAttention(embed, heads)
        self.mlp = MLP(mlp_layers)
        self.ln1, self.ln2 = nn.LayerNorm(embed), nn.LayerNorm(embed)

    def forward(self, x):
        att = x + self.attention(self.ln1(x))
        ff = att + self.mlp(self.ln2(att))
        return ff

# Embeds input, runs a # of transformer blocks, and a final linear output layer

class Transformer(nn.Module):
    def __init__(self, tokens, input_dim, embed_dim, output_dim, heads, mlp_layers, blocks):
        super().__init__()
        self.embed_matrix = nn.Linear(input_dim, embed_dim)
        self.positional = nn.Parameter(torch.randn(tokens, embed_dim))
        self.transformer = nn.Sequential()
        for i in range(blocks):
            self.transformer.add_module(f"block{i+1}", TransformerBlock(embed_dim, heads, mlp_layers))
        self.output = nn.Linear(embed_dim, output_dim)

    def embed(self, x):
        return self.embed_matrix(x) + self.positional # Assumes fixed token length

    def forward(self, x): # x = (B, tokens, input_dim)
        embedding = self.embed(x) # (B, tokens, embed_dim)
        transform = self.transformer(embedding)
        return self.output(transform).squeeze() # (B, tokens, output_dim)
    

# Baking in some explicit parameters

class ChessMLP(MLP):
    def __init__(self):
        super().__init__([192, 512, 256, 128, 64 * 13])

    def forward(self, x): # turns (B, 64 * 13) into (B, 64, 13)
        X = self.model(x) 
        return X.view(X.size(0), 64, 13).squeeze()

class ChessMLP_big(MLP):
    def __init__(self):
        super().__init__([192, 1536, 1024, 512, 64 * 13])

    def forward(self, x): # turns (B, 64 * 13) into (B, 64, 13)
        X = self.model(x) 
        return X.view(X.size(0), 64, 13).squeeze()

class ChessTransformer_small(Transformer):
    tokens = 64
    input_dim = 3
    def __init__(self):
        super().__init__(
            self.tokens, 
            self.input_dim, 
            embed_dim = 64, 
            output_dim = 13, 
            heads = 2, 
            mlp_layers = [64, 128, 64], 
            blocks = 2
        )

    def embed(self, x):
        x = x.view(x.size(0), self.tokens, self.input_dim)
        return super().embed(x)
    
class ChessTransformer_medium(Transformer):
    tokens = 64
    input_dim = 3
    def __init__(self):
        super().__init__(
            self.tokens, 
            self.input_dim, 
            embed_dim = 64, 
            output_dim = 13, 
            heads = 2, 
            mlp_layers = [64, 256, 64], 
            blocks = 4
        )

    def embed(self, x):
        x = x.view(x.size(0), self.tokens, self.input_dim)
        return super().embed(x)
    
class ChessTransformer_big(Transformer):
    tokens = 64
    input_dim = 3
    def __init__(self):
        super().__init__(
            self.tokens, 
            self.input_dim, 
            embed_dim = 128, 
            output_dim = 13, 
            heads = 2, 
            mlp_layers = [128, 512, 128], 
            blocks = 8
        )

    def embed(self, x):
        x = x.view(x.size(0), self.tokens, self.input_dim)
        return super().embed(x)