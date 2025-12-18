import torch
from chess_ml.train_utils import train_one_epoch
from chess_ml.model import MLP
from chess_ml.dataset import ChessDataset

num_epochs = 5 # adjust
layers = [192, 512, 256, 128, 64 * 13] # adjust
model = MLP(layers)
dataset = ChessDataset("data/train.pt", n_samples = 50000)
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