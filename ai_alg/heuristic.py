from game_structure import style as s
import numpy as np

def calculate_board_score(board: np.ndarray, piece: int, opponent_piece: int) -> int:
    score = 0

    # Verifica horizontal
    for col in range(s.COLUMNS - 3):
        for r in range(s.ROWS):
            segment = [board[r][col + i] for i in range(4)]
            score += weights(segment, piece, opponent_piece)

    # Verifica vertical
    for col in range(s.COLUMNS):
        for r in range(s.ROWS - 3):
            segment = [board[r + i][col] for i in range(4)]
            score += weights(segment, piece, opponent_piece)

    # Verifica diagonal ascendente
    for col in range(s.COLUMNS - 3):
        for r in range(s.ROWS - 3):
            segment = [board[r + i][col + i] for i in range(4)]
            score += weights(segment, piece, opponent_piece)

    # Verifica diagonal descendente
    for col in range(s.COLUMNS - 3):
        for r in range(3, s.ROWS):
            segment = [board[r - i][col + i] for i in range(4)]
            score += weights(segment, piece, opponent_piece)

    return score


def weights(segment: list, piece: int, opponent_piece: int) -> int:
    if piece in segment and opponent_piece in segment: return 0
    if segment.count(piece) == 1: return 1
    if segment.count(piece) == 2: return 10
    if segment.count(piece) == 3: return 50
    if segment.count(piece) == 4: return 1000
    if segment.count(opponent_piece) == 1: return -1
    if segment.count(opponent_piece) == 2: return -10
    if segment.count(opponent_piece) == 3: return -50
    if segment.count(opponent_piece) == 4: return -2000
    return 0
