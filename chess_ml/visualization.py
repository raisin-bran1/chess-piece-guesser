# To visualize our model's predictions

import chess
import tkinter as tk
import torch
from chess_ml.dataparser import Dataparser
import chess_ml.encoding as enc
import chess_ml.model as models

SQUARE_SIZE = 30

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

def draw_board(canvas, fen, colors = False): # draw board on tkinter canvas, colors = True to show only piece colors
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
    
    square_size = SQUARE_SIZE
    for rank in range(8):
        for file in range(8):
            x1, y1 = file * square_size, rank * square_size
            x2, y2 = x1 + square_size, y1 + square_size
            color = "#DDB88C" if (rank + file) % 2 == 0 else "#A66D4F"
            canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
            
            piece = board.piece_at(chess.square(file, 7 - rank))
            if piece:
                canvas.create_text(x1 + square_size/2, y1 + square_size/2, 
                                   text=UNICODE_PIECES[piece.symbol()], font=("Arial", SQUARE_SIZE * 7 // 12))

def create_canvas(root, name): # Creates board canvas in tkinter
    frame = tk.Frame(root)
    frame.pack(side=tk.LEFT, padx=10, pady=10)
    tk.Label(frame, text = name).pack()
    canvas1 = tk.Canvas(frame, width = SQUARE_SIZE * 8, height = SQUARE_SIZE * 8)
    canvas1.pack()
    return canvas1

def create_canvases(root, names): # Creates list of canvases from list of canvas names
    canvases = []
    for name in names:
        canvases.append(create_canvas(root, name))
    return canvases

def get_model_prediction(model, fen):
    with torch.inference_mode():
        return enc.output_to_fen(model(enc.fen_to_input(fen).float().unsqueeze(0)))

def draw_boards(canvas, fen, model1_pred, model2_pred):
    draw_board(canvas[0], fen, colors = True) # Colors only
    draw_board(canvas[1], model1_pred) # MLP Predictions
    draw_board(canvas[2], model2_pred) # Transformer Predictions
    draw_board(canvas[3], fen) # Real Position

def draw_random_boards(canvas, dataparser: Dataparser, model1, model2):
    fen = dataparser.generate_random_position()
    draw_boards(canvas, fen, get_model_prediction(model1, fen), get_model_prediction(model2, fen))

def view_tkinter(pgn_path, model1, model2, model1_path, model2_path):
    dataparser = Dataparser(pgn_path)

    state_dict = torch.load(f"models/{model1_path}")
    model1.load_state_dict(state_dict)
    model1.eval()

    state_dict = torch.load(f"models/{model2_path}")
    model2.load_state_dict(state_dict)
    model2.eval()

    root = tk.Tk()
    root.title("Boards")

    canvas = create_canvases(root, ["Colors", "MLP", "Transformer", "Real Board"])
    draw_random_boards(canvas, dataparser, model1, model2)

    root.bind("<Button-1>", lambda event: draw_random_boards(canvas, dataparser, model1, model2))

    root.mainloop()