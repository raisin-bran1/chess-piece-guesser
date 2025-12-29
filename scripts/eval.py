import torch
from chess_ml.evaluation import eval_model
import chess_ml.model as models
from chess_ml.dataset import ChessDataset

for epoch in range(5, 25, 5):
    eval = eval_model(
        model_path = f"transformer_{epoch}.pt",
        model = models.ChessTransformer(),
        dataset = ChessDataset("data/eval_2013-02.pt", n_samples = 100000),
        batch = 1000,
        device = torch.device("cpu")
    )
    print(eval)