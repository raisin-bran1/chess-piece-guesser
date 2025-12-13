import chess
import dataparser
import tkinter as tk

def convert_to_pawns(fen):
    new_fen = ''
    for char in fen:
        if char in 'rnbqk':
            new_fen += 'p'
        elif char in 'RNBQK':
            new_fen += 'P'
        else:
            new_fen += char
    return new_fen

def draw_board(canvas, fen, colors = False): # draw board on tkinter canvas, colors = True show only piece colors
    """Reusable drawing function"""
    UNICODE_PIECES = {
        'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
        'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙', None: ' '
    }
    if colors:
        UNICODE_PIECES = {
            'P': "⚪", 'p': "⚫", None: ' '
        }
        fen = convert_to_pawns(fen)
    board = chess.Board(fen)
    
    square_size = 60
    for rank in range(8):
        for file in range(8):
            x1, y1 = file * square_size, rank * square_size
            x2, y2 = x1 + square_size, y1 + square_size
            color = "#DDB88C" if (rank + file) % 2 == 0 else "#A66D4F"
            canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
            
            piece = board.piece_at(chess.square(file, 7 - rank))
            if piece:
                canvas.create_text(x1 + square_size/2, y1 + square_size/2, 
                                   text=UNICODE_PIECES[piece.symbol()], font=("Arial", 35))

def show_tkinter(fen):
    root = tk.Tk()
    root.title("Boards")

    # Create Frame 1 (Left)
    frame1 = tk.Frame(root)
    frame1.pack(side=tk.LEFT, padx=10, pady=10)
    tk.Label(frame1, text="Colors").pack()
    canvas1 = tk.Canvas(frame1, width=480, height=480)
    canvas1.pack()
    draw_board(canvas1, fen, colors = True)

    # Create Frame 2 (Right)
    frame2 = tk.Frame(root)
    frame2.pack(side=tk.LEFT, padx=10, pady=10)
    tk.Label(frame2, text="Pieces").pack()
    canvas2 = tk.Canvas(frame2, width=480, height=480)
    canvas2.pack()
    draw_board(canvas2, fen)

    root.mainloop()

# --- Usage ---
PGN_PATH = "data_processing/lichess_db_standard_rated_2013-01.pgn"
offsets = dataparser.parse_games(PGN_PATH)
fen_str = dataparser.generate_random_position(PGN_PATH, offsets)
show_tkinter(fen_str)