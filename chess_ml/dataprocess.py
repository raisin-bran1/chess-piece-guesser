# Turn PGN file into pytorch tensors
import chess.pgn

def download_training(pgn_path):
    pgn_file = open(pgn_path, encoding="utf-8")
    
    while True:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            break

    pgn_file.close()