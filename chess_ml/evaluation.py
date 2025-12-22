import torch

# Batch should evenly divide dataset size 
def eval_model(model_path, model, dataset, batch = 1000, device = torch.device("cpu")): 
    state_dict = torch.load(f"models/{model_path}")
    model.load_state_dict(state_dict)
    model.eval()
    eval_loader = torch.utils.data.DataLoader(dataset, batch_size = batch)

    total_boards = batch * len(eval_loader)
    correct_boards = 0
    total_squares = total_boards * 64
    correct_squares = 0
    total_perpiece = torch.zeros(13)
    correct_perpiece = torch.zeros(13)

    with torch.no_grad():
        for x, y in eval_loader:
            x = x.to(device).float()
            y = y.to(device)

            logits = model(x) # (B, 64, 13)
            predictions = logits.argmax(dim = -1) # (B, 64)
            correct = predictions == y # (B, 64) bool
            correct_preds = predictions[correct]  # (num_correct)

            correct_boards += correct.all(dim = 1).sum().item()
            correct_squares += correct_preds.size(0)
            total_perpiece += torch.unique(predictions, return_counts=True)[1]
            correct_perpiece += torch.unique(correct_preds, return_counts=True)[1]

    evaluation = f'''Boards: {(correct_boards / total_boards):.4f} ({correct_boards}/{total_boards})
Squares: {(correct_squares / total_squares):.4f} ({correct_squares}/{total_squares})
'''
    pieces = ['Empty'] + list('PRNBQKprnbqk')
    for i in range(13):
        correct, total = int(correct_perpiece[i].item()), int(total_perpiece[i].item())
        evaluation += f"{pieces[i]}: {(correct / total):.4f} ({correct}/{total})\n"
    
    return evaluation
