import torch
from chess_ml.evaluation import eval_model, plot_accuracy_barchart
import chess_ml.model as models
from chess_ml.dataset import ChessDataset

dataset = ChessDataset("data/eval_2013-02.pt", n_samples = 100000, eval = True)
batch = 1000
device = torch.device("cpu")

eval_mlp, bar_mlp = eval_model(
    model_path = f"mlp_basic.pt",
    model = models.ChessMLP(),
    dataset = dataset,
    batch = batch, 
    device = device
)
eval_tf, bar_tf = eval_model(
    model_path = f"transformer_basic.pt",
    model = models.ChessTransformer_basic(),
    dataset = dataset,
    batch = batch, 
    device = device
)
print("MLP\n" + eval_mlp)
print("Transformer\n" + eval_tf)

plot_accuracy_barchart(range(1, 41), bar_mlp[:40], bar_tf[:40], "evals/barchart.png")