import chess.pgn
import random

# Data pre-processing
def parse_games(pgn_path: str, filter = lambda headers: True): # filter function takes headers as input, returns something like headers.get("WhiteElo") >= 2000
    offsets = [] # List to store the byte offset of every game

    print("Scanning PGN for game offsets...")
    with open(pgn_path) as pgn:
        # read_headers is efficient; it skips moves and only looks at tags
        while True:
            offset = pgn.tell()
            headers = chess.pgn.read_headers(pgn)
            if headers is None:
                break
            if filter(headers):
                offsets.append(offset)
            
    if not offsets:
        print("No games found in file.")
        return []

    print(f"Found {len(offsets)} games.")
    return offsets

def get_random_game(pgn_path: str, offsets: list):
    random_offset = random.choice(offsets)

    with open(pgn_path) as pgn:
        # Seek (jump) directly to the start of that game
        pgn.seek(random_offset)
        
        # Read the full game (headers + moves)
        random_game = chess.pgn.read_game(pgn)
        
        return random_game
    
def get_positions(game: chess.pgn.Game):
    # Generates list of FEN positions from a game 
    positions = []
    board = game.board()
    for move in game.mainline_moves():
        positions.append(board.fen().split()[0])
        board.push(move)
    return positions

def generate_random_position(pgn_path: str, offsets: list):
    return random.choice(get_positions(get_random_game(pgn_path, offsets)))

