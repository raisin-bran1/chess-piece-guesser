from chess_ml.dataprocess import download_data

download_data("data/lichess_db_standard_rated_2013-02.pgn", "eval_filtered_2013-02.pt", move_filter = 10)