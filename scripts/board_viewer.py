from chess_ml.visualization import view_tkinter

def main():
    PGN_PATH = "data/lichess_db_standard_rated_2013-01.pgn"

    view_tkinter(PGN_PATH)

if __name__ == "__main__":
    main()