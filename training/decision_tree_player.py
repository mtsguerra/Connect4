import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_structure import style as s
from game_structure import game_engine as game
from training.decision_tree import ID3Tree

def board_to_feature_vector(board):
    """Converte o tabuleiro para um vetor de características"""
    
    return board.flatten()

def decision_tree_move(board):
    """Obtém a melhor jogada a partir do decision_tree"""
    
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "training", "data", "connect4_tree_model.pkl"
    )

    # Verifica se o modelo existe
    if not os.path.exists(model_path):
        print("Modelo de árvore de decisão não encontrado. Usando estratégia alternativa.")
        return fallback_strategy(board)

    try:
        model = ID3Tree.load(model_path)
        features = np.array([board_to_feature_vector(board)])
        move = model.predict(features)[0]
        # Garante que o movimento é int e está em um intervalo adequado
        try:
            move = int(move)
        except Exception:
            print(f"Movimento previsto {move} não pôde ser convertido para inteiro. Usando alternativa.")
            return fallback_strategy(board)
        if 0 <= move < board.shape[1] and is_valid_move(board, move):
            return move
        else:
            print(f"Movimento inválido {move} da árvore de decisão. Usando alternativa.")
            return fallback_strategy(board)
    except Exception as e:
        print(f"Erro ao usar o modelo de árvore de decisão: {e}")
        return fallback_strategy(board)

def is_valid_move(board, col):
    if col is None or not isinstance(col, (int, np.integer)):
        return False
    if col < 0 or col >= board.shape[1]:
        return False
    return board[0][col] == 0

def fallback_strategy(board):
    # Tenta a coluna central primeiro, depois as colunas adjacentes
    preferred_cols = [3, 2, 4, 1, 5, 0, 6]
    for col in preferred_cols:
        if is_valid_move(board, col):
            return col
    # Alternativa: primeira coluna disponível
    for col in range(board.shape[1]):
        if is_valid_move(board, col):
            return col
    # Nenhum movimento válido
    return 0  # Em vez de -1, sempre retorna 0 (alternativa segura)

def get_next_open_row(board, col):
    for r in range(board.shape[0] - 1, -1, -1):
        if board[r][col] == 0:
            return r
    return -1

def is_winning_move(board, piece):
    from game_structure import game_engine as game
    return game.winning_move(board, piece)

if __name__ == "__main__":
    board = np.zeros((s.ROWS, s.COLUMNS), dtype=int)
    move = decision_tree_move(board)
    print(f"A árvore de decisão sugere a coluna {move+1} para o tabuleiro vazio")
