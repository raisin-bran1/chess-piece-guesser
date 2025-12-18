# Turn PGN file into pytorch tensors
import chess.pgn
import chess_ml.encoding as enc
import torch
from datetime import datetime

def download_data(pgn_path, save_name = "train.pt"):
    pgn_file = open(pgn_path, encoding="utf-8")

    def get_positions(game: chess.pgn.Game):
        # Generates list of FEN positions from a game 
        positions = []
        board = game.board()
        for move in game.mainline_moves():
            positions.append(board.fen().split()[0])
            board.push(move)
        return positions
    
    print(f"Downloading {pgn_path}")
    input_tensors, target_tensors = [], []
    while True:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            break
        for position in get_positions(game):
            input_tensors.append(enc.fen_to_input(position))
            target_tensors.append(enc.fen_to_target(position))
    print(f"{len(input_tensors)} positions downloaded")
    pgn_file.close()

    data = {
        "inputs": torch.stack(input_tensors),
        "targets": torch.stack(target_tensors),
        "created_at": datetime.now()
    }

    torch.save(data, f"data/{save_name}")
    print(f"Data saved to data/{save_name}")
