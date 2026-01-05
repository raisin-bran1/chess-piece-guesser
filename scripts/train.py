import torch
from chess_ml.train_utils import train_model
import chess_ml.model as models
from chess_ml.dataset import ChessDataset

train_model(
    model_name = "transformer_alldata",
    model = models.ChessTransformer_medium(),
    dataset = ChessDataset("data/train_2013-01.pt", n_samples = None),
    num_epochs = 5,
    batch = 256,
    learning_rate = 1e-3,
    device = torch.device("cpu")
)