from game_structure import style as s
import numpy as np
import math
import pygame
from game_structure import Board
from ai_alg import basic_heuristic as b, alpha_beta as a, monte_carlo as m
from training.decision_tree_player import decision_tree_move

def first_player_move(bd: Board, interface: any, board: np.ndarray, turn: int, event: any) -> bool:
    """Define a coluna jogada pelo jogador 1"""
    
    col = get_human_column(interface, event)
    if not is_valid(board, col): return False

    pygame.draw.rect(interface.screen, s.BACKGROUND_COLOR, (0, 0, interface.width, interface.pixels - 14))
    make_move(bd, interface, board, turn, col)
    return True

def get_human_column(interface: any, event: any):
    """Obtém a coluna selecionada pelo mouse"""
    
    posx = event.pos[0]
    col = int(math.floor(posx / interface.pixels)) - 2
    return col

def available_moves(board: np.ndarray) -> list | int:
    """Retorna uma lista de colunas disponíveis para jogar, ou -1 se não houver"""
    
    avaiable_moves = []
    for i in range(s.COLUMNS):
        if (board[5][i]) == 0:
            avaiable_moves.append(i)
    return avaiable_moves if len(avaiable_moves) > 0 else -1

def ai_move(bd: Board, interface: any, game_mode: int, board: np.ndarray, turn: int) -> int:
    """Define a coluna jogada pelo jogador 2"""
    
    ai_column = get_ai_column(board, game_mode, turn)
    game_over = make_move(bd, interface, board, turn, ai_column)
    return game_over

def get_ai_column(board: Board, game_mode: int, player: int = 2) -> int:
    """Seleciona o algoritmo de IA escolhido para jogar"""
    
    opponent = 1 if player == 2 else 2

    if game_mode == 2:
        return b.evaluate_best_move(board, player, opponent)
    elif game_mode == 3:
        return m.mcts(board)
    elif game_mode == 4:
        return a.alpha_beta(board)
    elif game_mode == 5:
        return decision_tree_move(board)
    return 0

def simulate_move(board: np.ndarray, piece: int, col: int) -> np.ndarray:
    """Simula uma jogada em uma cópia do tabuleiro"""
    
    board_copy = board.copy()
    row = get_next_open_row(board_copy, col)
    drop_piece(board_copy, row, col, piece)
    return board_copy

def make_move(bd: Board, interface: any, board: np.ndarray, turn: int, move: int):
    """Executa a jogada e verifica se ela resulta em vitória ou empate"""

    row = get_next_open_row(board, move)
    drop_piece(board, row, move, turn) 
    interface.draw_new_piece(row + 1, move + 2, turn)
    pygame.display.update()
    bd.print_board()

    return winning_move(board, turn) or is_game_tied(board)

def get_next_open_row(board: np.ndarray, col: int) -> int:
    """Retorna a linha disponível para colocar a peça na coluna escolhida"""
    
    for row in range(s.ROWS):
        if board[row, col] == 0:
            return row
    return -1

def drop_piece(board: np.ndarray, row: int, col: int, piece: int) -> None:
    """Insere a peça no tabuleiro na posição escolhida"""
    
    board[row, col] = piece

def is_game_tied(board: np.ndarray) -> bool:
    """Verifica se o jogo empatou"""
    
    if winning_move(board, s.SECOND_PLAYER_PIECE) or winning_move(board, s.FIRST_PLAYER_PIECE): return False
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0: return False
    return True

def is_valid(board: np.ndarray, col: int) -> bool:
    """Analisa se a coluna escolhida é válida"""
    
    if not 0 <= col < s.COLUMNS: return False
    row = get_next_open_row(board, col)
    return 0 <= row <= 5

def winning_move(board: np.ndarray, piece: int) -> bool:
    """Analisa se a peça colocada resultou em vitória"""
    
    rows, cols = board.shape
    for row in range(rows):
        for col in range(cols):
            if int(board[row, col]) == piece:
                # Checa horizontalmente
                if col + 3 < cols and all(board[row, col + i] == piece for i in range(4)):
                    return True
                # Checa verticalmente
                if row + 3 < rows and all(board[row + i, col] == piece for i in range(4)):
                    return True
                # Checa diagonal ascendente
                if row - 3 >= 0 and col + 3 < cols and all(board[row - i, col + i] == piece for i in range(4)):
                    return True
                # Checa diagonal descendente
                if row + 3 < rows and col + 3 < cols and all(board[row + i, col + i] == piece for i in range(4)):
                    return True
    return False
