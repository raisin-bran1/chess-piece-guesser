# Turn PGN file into pytorch tensors
import chess.pgn
import chess_ml.encoding as enc
import torch
from datetime import datetime

def get_positions(game: chess.pgn.Game):
    # Generates list of FEN positions from a game 
    positions = []
    board = game.board()
    for move in game.mainline_moves():
        positions.append(board.fen().split()[0])
        board.push(move)
    return positions

def download_data(pgn_path, save_name = "train.pt", move_filter = 0):
    pgn_file = open(pgn_path, encoding="utf-8")
    
    print(f"Downloading {pgn_path}")
    input_tensor, target_tensor, move_tensor = [], [], []
    while True:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            break
        for move, position in enumerate(get_positions(game)[move_filter:], start = 1):
            input_tensor.append(enc.fen_to_input(position))
            target_tensor.append(enc.fen_to_target(position))
            move_tensor.append(move)
    print(f"{len(input_tensor)} positions downloaded")
    pgn_file.close()

    data = {
        "inputs": torch.stack(input_tensor),
        "targets": torch.stack(target_tensor),
        "moves": torch.tensor(move_tensor),
        "created_at": datetime.now()
    }

    torch.save(data, f"data/{save_name}")
    print(f"Data saved to data/{save_name}")
