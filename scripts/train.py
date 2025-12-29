import torch
from chess_ml.train_utils import train_model
import chess_ml.model as models
from chess_ml.dataset import ChessDataset

train_model(
    model_name = "transformer",
    model = models.ChessTransformer(),
    dataset = ChessDataset("data/train_2013-01.pt", n_samples = 100000),
    num_epochs = 20,
    batch = 128,
    learning_rate = 1e-3,
    device = torch.device("cpu")
)