import chess_ml.encoding as enc

fen = '1Bn4q/3B4/5r2/b1R1Q1K1/1p6/1P1pP2p/P4k2/8'
encoding = enc.fen_to_target(fen)
print(encoding)
