import torch
from chess_ml.evaluation import eval_model
import chess_ml.model as models
from chess_ml.dataset import ChessDataset

eval = eval_model(
    model_path = f"mlp_alldata.pt",
    model = models.ChessMLP(),
    dataset = ChessDataset("data/eval_2013-02.pt", n_samples = 100000),
    batch = 1000,
    device = torch.device("cpu")
)
print(eval)