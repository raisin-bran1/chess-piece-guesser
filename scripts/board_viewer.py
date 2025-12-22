from chess_ml.visualization import view_tkinter
import chess_ml.model as models

def main():
    PGN_PATH = "data/lichess_db_standard_rated_2013-02.pgn"
    MODEL = models.MLP_basic()
    MODEL_PATH = "mlp_basic.pt"

    view_tkinter(PGN_PATH, MODEL, MODEL_PATH)

if __name__ == "__main__":
    main()