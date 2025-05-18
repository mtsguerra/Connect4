import numpy as np
from game_structure import style as s
from game_structure import game_engine as game
from ai_alg import heuristic as h


def alpha_beta(board: np.ndarray) -> int:
    """Usando o algoritmo de alpha-beta, escolhe a melhor jogada até uma certa profundidade"""
    max_depth = 5
    best_col = -1
    best_score = float('-inf')

    for new_board, col in generate_children(board, s.SECOND_PLAYER_PIECE):
        if game.winning_move(new_board, s.SECOND_PLAYER_PIECE):
            return col
        score = evaluate_position(
            new_board, current_depth=1,
            alpha=float('-inf'), beta=float('inf'),
            depth_limit=max_depth, is_maximizing=False
        )
        if score > best_score:
            best_score = score
            best_col = col

    return best_col


def evaluate_position(board: np.ndarray,current_depth: int,alpha: float,beta: float,depth_limit: int,is_maximizing: bool) -> float:
    """Usa minimax e alpha-beta para avaliar o tabuleiro"""
    if (current_depth == depth_limit or game.winning_move(board, s.FIRST_PLAYER_PIECE) or game.winning_move(board, s.SECOND_PLAYER_PIECE) or game.is_game_tied(board)):
        return h.calculate_board_score(board, s.SECOND_PLAYER_PIECE, s.FIRST_PLAYER_PIECE)

    if is_maximizing:
        max_eval = float('-inf')
        for child_board, _ in generate_children(board, s.SECOND_PLAYER_PIECE):
            evaluation = evaluate_position(child_board, current_depth + 1, alpha, beta, depth_limit, False)
            max_eval = max(max_eval, evaluation)
            alpha = max(alpha, evaluation)
            if beta <= alpha:
                break
        return max_eval

    else:
        min_eval = float('inf')
        for child_board, _ in generate_children(board, s.FIRST_PLAYER_PIECE):
            evaluation = evaluate_position(child_board, current_depth + 1, alpha, beta, depth_limit, True)
            min_eval = min(min_eval, evaluation)
            beta = min(beta, evaluation)
            if beta <= alpha:
                break
        return min_eval


def generate_children(board: np.ndarray, piece: int):
    """Gera tabuleiros filhos para uma determinada peça."""
    children = []
    available = game.available_moves(board)
    if available == -1:
        return children
    for col in available:
        new_board = game.simulate_move(board, piece, col)
        children.append((new_board, col))
    return children
