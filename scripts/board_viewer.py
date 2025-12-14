import chess_ml.visualization

def main():
    PGN_PATH = "data/lichess_db_standard_rated_2013-01.pgn"

    chess_ml.visualization.view_tkinter(PGN_PATH)

if __name__ == "__main__":
    main()