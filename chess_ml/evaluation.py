import torch
from collections import Counter
import matplotlib.pyplot as plt

def resize_input(x):
    return x.view(x.size(0), 64, 3).argmax(dim = -1) # (B, 64)

def is_regular_pieces(tuple): # For a board tuple, all piece counts <= upper bounds and kings exist
    freq = Counter(tuple)
    upper_bounds = [64] + 2 * [8, 2, 2, 2, 1, 1]
    compare = zip([freq[i] for i in range(13)], upper_bounds)
    return all([a <= b for a, b in compare]) and freq[6] and freq[12] 

def eval_string(title, correct, total):
    return f"{title}: {(correct / total):.4f} ({correct}/{total})\n"

# Batch should evenly divide dataset size 
# This really should've been implemented as a class huh
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
    proper_colors = 0
    proper_pieces = 0
    both = 0
    square_acc = torch.zeros(64)
    correct_bymove, total_bymove = torch.zeros(dataset.max_moves), torch.zeros(dataset.max_moves)

    with torch.no_grad():
        for x, y, moves in eval_loader:
            x = x.to(device).float() # (B, 192)
            y = y.to(device) # (B, 64)
            moves = moves.to(device) # (B)

            logits = model(x) # (B, 64, 13)
            predictions = logits.argmax(dim = -1) # (B, 64)
            correct = predictions == y # (B, 64) bool
            correct_rows = correct.all(dim = 1) # (B)
            correct_preds = predictions[correct]  # (num_correct)
            color_match = (resize_input(x) == torch.ceil(predictions / 6)).all(dim = 1) # (B)
            regular_pieces = torch.tensor([is_regular_pieces(row.tolist()) for row in predictions], dtype=torch.bool) # (B)

            correct_boards += correct_rows.sum().item()
            correct_squares += correct_preds.size(0)
            total_perpiece += torch.unique(predictions, return_counts=True)[1]
            correct_perpiece += torch.unique(correct_preds, return_counts=True)[1]
            proper_colors += color_match.sum().item()
            proper_pieces += regular_pieces.sum().item()
            both += torch.logical_and(color_match, regular_pieces).sum().item()
            square_acc += torch.sum(correct, dim = 0)

            # correct boards by move count
            for i in range(batch):
                total_bymove[moves[i]-1] += 1
                correct_bymove[moves[i]-1] += correct_rows[i]
    
    correct_pieces = correct_squares - int(correct_perpiece[0].item())
    total_pieces = total_squares - int(total_perpiece[0].item())
    square_acc /= total_boards
    acc_bymove = correct_bymove / total_bymove

    evaluation = (eval_string("Boards", correct_boards, total_boards) + 
        eval_string("Squares", correct_squares, total_squares) + 
        eval_string("Pieces", correct_pieces, total_pieces))

    pieces = ['Empty'] + list('PRNBQKprnbqk')
    for i in range(13):
        correct, total = int(correct_perpiece[i].item()), int(total_perpiece[i].item())
        evaluation += eval_string(pieces[i], correct, total)
    evaluation += (eval_string("Correct colors", proper_colors, total_boards) + 
        eval_string("Regular pieces", proper_pieces, total_boards) +
        eval_string("Both", both, total_boards))
    
    plot_heatmap(square_acc, model.name, f"evals/{model_path[:-3]}_heatmap.png")
    return evaluation, acc_bymove


def plot_accuracy_barchart(
    x, 
    acc_mlp,
    acc_tf,  
    filename, 
):
    fig, ax = plt.subplots()
    width = 0.4
    ax.bar([i - width/2 for i in x], acc_mlp, width, label='MLP')
    ax.bar([i + width/2 for i in x], acc_tf, width, label='Transformer')

    ax.set_xlabel("Move")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy by move count")
    ax.legend()

    plt.savefig(filename)
    

def plot_heatmap(
    x,
    modelname,
    filename,
    cmap="viridis",
    fmt="{:.2f}",
    dpi=300
):
    """
    Plot and save an n x n heatmap from a 1D torch tensor of length n^2.

    Parameters
    ----------
    x : torch.Tensor of shape (n*n,)
        Input 1D tensor.
    filename : str
        Output image filename.
    cmap : str
        Matplotlib colormap.
    fmt : str
        Format string for values.
    dpi : int
        Resolution of saved image.
    """
    assert x.ndim == 1, "Input must be a 1D tensor"
    n2 = x.numel()
    n = int(n2 ** 0.5)
    assert n * n == n2, "Length of tensor must be a perfect square"

    # Reshape and move to CPU for matplotlib
    A = x.view(n, n).detach().cpu()

    fig, ax = plt.subplots()
    im = ax.imshow(A, cmap=cmap, vmin = 0.85, vmax = 1)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Accuracy", rotation=-90, va="bottom")

    # Ticks
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    ax.set_yticklabels(range(n, 0, -1))

    # Annotate cells
    for i in range(n):
        for j in range(n):
            value = A[i, j].item()
            ax.text(
                j, i, fmt.format(value),
                ha="center",
                va="center",
                color="white" if im.norm(value) > 0.5 else "black"
            )

    ax.set_title(f"{modelname} Heatmap")
    fig.tight_layout()

    plt.savefig(filename, dpi=dpi, bbox_inches="tight")