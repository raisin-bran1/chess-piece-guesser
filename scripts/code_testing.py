import chess_ml.encoding as enc

fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'
enc.fen_to_input(fen)
enc.fen_to_target(fen)