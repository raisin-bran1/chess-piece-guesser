import torch
from chess_ml.train_utils import train_one_epoch
import chess_ml.model as models
from chess_ml.dataset import ChessDataset

num_epochs = 5 # adjust
model = models.MLP_basic() # adjust layers
dataset = ChessDataset("data/train.pt", n_samples = 50000) # adjust n_samples
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size = 64, # adjust
    shuffle=True
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # adjust lr
device = torch.device("cpu")

print("Starting training!")
for epoch in range(num_epochs):
    loss = train_one_epoch(model, train_loader, optimizer, device)
    print(f"Epoch {epoch + 1}: loss = {loss:.4f}")

model_name = "mlp_basic.pt"
torch.save(model.state_dict(), f"models/{model_name}")

# Implement way to document hyperparameters for each trained model