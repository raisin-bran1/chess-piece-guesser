from chess_ml.visualization import view_tkinter
import chess_ml.model as models

def main():
    PGN_PATH = "data/lichess_db_standard_rated_2013-02.pgn"
    MLP = models.ChessMLP_big()
    TRANSFORMER = models.ChessTransformer()
    MLP_PATH = "mlp_big.pt"
    TRANSFORMER_PATH = "transformer.pt"

    view_tkinter(PGN_PATH, MLP, TRANSFORMER, MLP_PATH, TRANSFORMER_PATH)

if __name__ == "__main__":
    main()