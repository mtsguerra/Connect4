import pygame
import numpy as np
from game_structure import Board
from game_structure import style as s
from game_structure import game_engine as game
from ai_alg import basic_heuristic as b, alpha_beta as a, monte_carlo as m, heuristic as h
from training.decision_tree_player import decision_tree_move  # <-- Add this import
import itertools
import time

def run_ai_vs_ai_game(interface, brd, game_mode):
    """
    Run a game between two AI players based on the selected game mode
    """
    board = brd.get_board()
    game_over = False
    turns = itertools.cycle([1, 2])  # Player 1 and Player 2
    turn = next(turns)
    
    # Set up the AI types for each player based on game mode
    ai_types = get_ai_types(game_mode)
    
    interface.draw_board()
    pygame.display.update()
    
    # Small delay before starting
    pygame.time.wait(1000)
    
    move_count = 0
    
    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                interface.quit()
                
        # Add a small delay between moves to make the game watchable
        pygame.time.wait(500)
        
        # Get AI column based on the current player and their assigned AI type
        ai_type = ai_types[turn - 1]  # Adjust for 0-based indexing
        col = get_ai_column_for_type(board, turn, ai_type)
        
        # Make the move and check if game is over
        row = game.get_next_open_row(board, col)
        if row != -1:  # Valid move
            game.drop_piece(board, row, col, turn)
            interface.draw_new_piece(row + 1, col + 2, turn)
            pygame.display.update()
            brd.print_board()
            
            move_count += 1
            print(f"Move {move_count}: Player {turn} ({ai_type}) placed at column {col}")
            
            # Check if this move wins the game
            if game.winning_move(board, turn):
                font = pygame.font.SysFont('Connect4-main/fonts/SuperMario256.ttf', 50)
                interface.show_winner(font, turn)
                game_over = True
                break
                
            # Check if the game is tied
            if game.is_game_tied(board):
                font = pygame.font.SysFont('Connect4-main/fonts/SuperMario256.ttf', 50)
                interface.show_draw(font)
                game_over = True
                break
                
        # Switch to the next player
        turn = next(turns)
    
    # Wait before closing
    pygame.time.wait(10000)
    return

def get_ai_types(game_mode):
    """
    Returns the AI types for both players based on the game mode
    """
    ai_mapping = {
        6: ["Easy", "Easy"],           # Easy x Easy
        7: ["Easy", "Hard"],           # Easy x Hard
        8: ["Medium", "Medium"],       # Medium x Medium (now MCTS)
        9: ["Medium", "Challenge"],    # MCTS x DecisionTree
        10: ["Hard", "Challenge"],     # AlphaBeta x DecisionTree
        11: ["Easy", "Medium"],        # Easy x MCTS
        12: ["Easy", "Challenge"],     # Easy x DecisionTree
        13: ["Medium", "Hard"],        # MCTS x AlphaBeta
        14: ["Hard", "Hard"],          # AlphaBeta x AlphaBeta
        15: ["Challenge", "Challenge"] # DecisionTree x DecisionTree
    }
    return ai_mapping.get(game_mode, ["Easy", "Easy"])

def get_ai_column_for_type(board, player, ai_type):
    """
    Gets the best move for the specified AI type
    """
    opponent = 1 if player == 2 else 2
    
    if ai_type == "Easy":
        # Use basic heuristic (A*)
        return b.evaluate_best_move(board, player, opponent)
    elif ai_type == "Medium":
        # Use Monte Carlo Tree Search (MCTS)
        return m.mcts(board)
    elif ai_type == "Hard":
        # Use Alpha-Beta pruning
        return a.alpha_beta(board)
    elif ai_type == "Challenge":
        # Use Decision Tree agent
        return decision_tree_move(board)
    else:
        # Default to basic heuristic
        return b.evaluate_best_move(board, player, opponent)
