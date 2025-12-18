# Training Loop
import torch
import torch.nn.functional as F

def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for x, y in train_loader:
        x = x.to(device).float()   # (B, 192)
        y = y.to(device).long()   # (B, 64)

        optimizer.zero_grad()

        logits = model(x)  # (B, 64, 13)

        loss = F.cross_entropy(
            logits.view(-1, 13),   # (B*64, 13)
            y.view(-1)             # (B*64)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)