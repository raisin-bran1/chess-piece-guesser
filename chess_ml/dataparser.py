# Skims pgn file for random position retrieval, used for visualization only  

import chess.pgn
import random

class Dataparser:
    def __init__(self, pgn_path: str, filter = lambda headers: True): # filter function takes headers as input, returns something like headers.get("WhiteElo") >= 2000
        self.pgn_path = pgn_path
        self.offsets = [] # List to store the byte offset of every game

        print("Scanning PGN for game offsets...")
        with open(self.pgn_path) as pgn:
            # read_headers is efficient; it skips moves and only looks at tags
            while True:
                offset = pgn.tell()
                headers = chess.pgn.read_headers(pgn)
                if headers is None:
                    break
                if filter(headers):
                    self.offsets.append(offset)
                
        print(f"Found {len(self.offsets)} games.")
        self.len = len(self.offsets)

    def get_random_game(self):
        random_offset = random.choice(self.offsets)

        with open(self.pgn_path) as pgn:
            # Seek (jump) directly to the start of that game
            pgn.seek(random_offset)
            
            # Read the full game (headers + moves)
            random_game = chess.pgn.read_game(pgn)
            
            return random_game
        
    def get_positions(self, game: chess.pgn.Game):
        # Generates list of FEN positions from a game 
        positions = []
        board = game.board()
        for move in game.mainline_moves():
            positions.append(board.fen().split()[0])
            board.push(move)
        return positions

    def generate_random_position(self):
        return random.choice(self.get_positions(self.get_random_game()))

