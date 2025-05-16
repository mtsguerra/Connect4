import numpy as np
from game_structure import style as s
from game_structure import game_engine as game
from ai_alg import heuristic as h


def evaluate_best_move(board: np.ndarray, player_piece: int, enemy_piece: int) -> int:
    """
    Evaluate all legal moves and return the one that gives the highest score
    for the current player using the heuristic function.
    """
    top_score = float('-inf')
    chosen_column = -1

    for column in game.available_moves(board):
        future_board = game.simulate_move(board, player_piece, column)
        evaluation = h.calculate_board_score(future_board, player_piece, enemy_piece)

        if evaluation > top_score:
            top_score = evaluation
            chosen_column = column

    return chosen_column


def adversarial_lookahead(board: np.ndarray, player_piece: int, enemy_piece: int) -> int:
    """
    Consider both the player's move and the opponent's most optimal response,
    then choose the move that results in the least damage or best advantage.
    """
    optimal_column = -1
    highest_evaluation = float('-inf')
    suggested_counter = 0

    legal_columns = game.available_moves(board)
    if len(legal_columns) == 1:
        return legal_columns[0]

    for column in legal_columns:
        player_future = game.simulate_move(board, player_piece, column)

        # If the move results in an immediate win, take it
        if game.winning_move(player_future, s.SECOND_PLAYER_PIECE):
            return column

        # Predict opponent’s optimal move and simulate it
        predicted_opponent = evaluate_best_move(player_future, enemy_piece, player_piece)
        opponent_future = game.simulate_move(player_future, enemy_piece, predicted_opponent)

        evaluation = h.calculate_board_score(opponent_future, player_piece, enemy_piece)

        if evaluation > highest_evaluation:
            highest_evaluation = evaluation
            optimal_column = column
            suggested_counter = predicted_opponent + 1

    print("Dica de jogada: coluna", suggested_counter + 1)
    return optimal_column
