import pygame
import numpy as np
from game_structure import Board
from game_structure import style as s
from game_structure import game_engine as game
from ai_alg import basic_heuristic as b, alpha_beta as a, monte_carlo as m, heuristic as h
from training.decision_tree_player import decision_tree_move  # <-- Mantenha este import
import itertools
import time

def run_ai_vs_ai_game(interface, brd, game_mode):
    """Simula uma partida entre duas IA, podendo ou não serem diferentes"""
    
    board = brd.get_board()
    game_over = False
    turns = itertools.cycle([1, 2]) 
    turn = next(turns)
    
    # Define qual IA cada jogador será
    ai_types = get_ai_types(game_mode)
    
    interface.draw_board()
    pygame.display.update()
    
    pygame.time.wait(1000)
    
    move_count = 0
    
    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                interface.quit()
                
        # Pequeno atraso entre as jogadas para as partidas poderem serem assistidas
        pygame.time.wait(500)
        
        ai_type = ai_types[turn - 1]
        col = get_ai_column_for_type(board, turn, ai_type)
        
        row = game.get_next_open_row(board, col)
        if row != -1:
            game.drop_piece(board, row, col, turn)
            interface.draw_new_piece(row + 1, col + 2, turn)
            pygame.display.update()
            brd.print_board()
            move_count += 1
            print(f"Jogada {move_count}: Jogador {turn} ({ai_type}) colocou na coluna {col}")
            
            if game.winning_move(board, turn):
                font = pygame.font.SysFont('Connect4-main/fonts/SuperMario256.ttf', 50)
                interface.show_winner(font, turn)
                game_over = True
                break  
            if game.is_game_tied(board):
                font = pygame.font.SysFont('Connect4-main/fonts/SuperMario256.ttf', 50)
                interface.show_draw(font)
                game_over = True
                break
                
        turn = next(turns)
    
    pygame.time.wait(10000)
    return

def get_ai_types(game_mode):
    """Retorna as IAs escolhidas"""
    
    ai_mapping = {
        6: ["Easy", "Easy"],           # Fácil x Fácil
        7: ["Easy", "Hard"],           # Fácil x Difícil
        8: ["Medium", "Medium"],       # Médio x Médio
        9: ["Medium", "Challenge"],    # Médio x Desafio
        10: ["Hard", "Challenge"],     # AlphaBeta x Desafio
        11: ["Easy", "Medium"],        # Fácil x Médio
        12: ["Easy", "Challenge"],     # Fácil x Desafio
        13: ["Medium", "Hard"],        # Médio x AlphaBeta
        14: ["Hard", "Hard"],          # AlphaBeta x AlphaBeta
        15: ["Challenge", "Challenge"] # Desafio x Desafio
    }
    return ai_mapping.get(game_mode, ["Easy", "Easy"])

def get_ai_column_for_type(board, player, ai_type):
    """Obtém a melhor jogada de acordo com a respectiva IA"""
    
    opponent = 1 if player == 2 else 2
    
    if ai_type == "Easy":
        # Usa heurística básica (A*)
        return b.evaluate_best_move(board, player, opponent)
    elif ai_type == "Medium":
        # Usa Monte Carlo Tree Search (MCTS)
        return m.mcts(board)
    elif ai_type == "Hard":
        # Usa poda Alpha-Beta
        return a.alpha_beta(board)
    elif ai_type == "Challenge":
        # Usa agente Decision Tree
        return decision_tree_move(board)
    else:
        # Padrão para heurística básica
        return b.evaluate_best_move(board, player, opponent)
