import numpy as np
from game_structure import style as s
from game_structure import game_engine as game
from ai_alg import heuristic as h


def evaluate_best_move(board: np.ndarray, player_piece: int, enemy_piece: int) -> int:
    """Avalia todas as possíveis jogadas e retorna aquela com a maior pontuação, utilizando a função heurística"""
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
    """Considera a próxima jogada e a melhor repostas possível do adversário a esta resposta, escolhendo assim a jogada com melhor vantagem"""
    optimal_column = -1
    highest_evaluation = float('-inf')
    suggested_counter = 0

    legal_columns = game.available_moves(board)
    if len(legal_columns) == 1:
        return legal_columns[0]

    for column in legal_columns:
        player_future = game.simulate_move(board, player_piece, column)
        if game.winning_move(player_future, s.SECOND_PLAYER_PIECE):
            return column
        predicted_opponent = evaluate_best_move(player_future, enemy_piece, player_piece)
        opponent_future = game.simulate_move(player_future, enemy_piece, predicted_opponent)
        evaluation = h.calculate_board_score(opponent_future, player_piece, enemy_piece)
        if evaluation > highest_evaluation:
            highest_evaluation = evaluation
            optimal_column = column
            suggested_counter = predicted_opponent + 1

    print("Possivel jogada: coluna no", suggested_counter + 1)
    return optimal_column
