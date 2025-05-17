import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_structure import style as s
from game_structure import game_engine as game
from training.decision_tree import ID3Tree

def board_to_feature_vector(board):
    """Convert board to feature vector"""
    return board.flatten()

def decision_tree_move(board):
    """Get best move from decision tree model""" 
    # Path to saved model
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "training", "data", "connect4_tree_model.pkl"
    )
    
    # Check if model exists
    if not os.path.exists(model_path):
        print("Decision tree model not found. Using fallback strategy.")
        return fallback_strategy(board)
    
    try:
        # Load model
        model = ID3Tree.load(model_path)
        
        # Convert board to feature vector
        features = np.array([board_to_feature_vector(board)])
        
        # Get prediction
        move = model.predict(features)[0]
        
        # Validate move
        if is_valid_move(board, move):
            return move
        else:
            print(f"Invalid move {move} from decision tree. Using fallback.")
            return fallback_strategy(board)
            
    except Exception as e:
        print(f"Error using decision tree model: {e}")
        return fallback_strategy(board)

def is_valid_move(board, col):
    """Check if a move is valid"""
    if col < 0 or col >= board.shape[1]:
        return False
    return board[0][col] == 0

def fallback_strategy(board):
    """Simple strategy when decision tree fails"""
    # Try center column first, then adjacent columns
    preferred_cols = [3, 2, 4, 1, 5, 0, 6]
    
    # Check for winning move
    for col in range(board.shape[1]):
        if not is_valid_move(board, col):
            continue
            
        # Try move
        temp_board = board.copy()
        row = get_next_open_row(temp_board, col)
        temp_board[row][col] = s.SECOND_PLAYER_PIECE
        
        if is_winning_move(temp_board, s.SECOND_PLAYER_PIECE):
            return col
    
    # Check for blocking opponent's win
    for col in range(board.shape[1]):
        if not is_valid_move(board, col):
            continue
            
        # Try move for opponent
        temp_board = board.copy()
        row = get_next_open_row(temp_board, col)
        temp_board[row][col] = s.FIRST_PLAYER_PIECE
        
        if is_winning_move(temp_board, s.FIRST_PLAYER_PIECE):
            return col
    
    # Use preferred column order
    for col in preferred_cols:
        if is_valid_move(board, col):
            return col
            
    # Fallback to first available column
    for col in range(board.shape[1]):
        if is_valid_move(board, col):
            return col
            
    # No valid moves
    return -1

def get_next_open_row(board, col):
    """Find the next available row in a column"""
    for r in range(board.shape[0]-1, -1, -1):
        if board[r][col] == 0:
            return r
    return -1

def is_winning_move(board, piece):
    """Check if the last move was a winning move"""
    # This is a simplified version, you should use your actual winning check function
    from game_structure import game_engine as game
    return game.winning_move(board, piece)

if __name__ == "__main__":
    # Test the decision tree player
    board = np.zeros((s.ROWS, s.COLUMNS), dtype=int)
    move = decision_tree_move(board)
    print(f"Decision tree suggests column {move+1} for empty board")