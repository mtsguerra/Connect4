import csv
import os
import sys
import time
import numpy as np
from multiprocessing import Pool, cpu_count
from typing import Optional, List, Tuple

# Add the root directory to the path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_structure import style as s
from game_structure import game_engine as game
from ai_alg.monte_carlo_ts import MonteCarlo, Node

# Directory where CSV will be saved
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def board_to_feature_vector(board: np.ndarray) -> List[int]:
    """
    Converts the Connect4 board to a flattened feature vector.
    The board is 6x7, resulting in a vector of length 42.
    """
    # Flatten the board into a 1D array
    return board.flatten().tolist()

def generate_game(params: Tuple[int, int]) -> List[List[int]]:
    """
    Simulates a Connect4 game using Monte Carlo Tree Search.
    Returns a list of state-action pairs, where each state is a flattened board
    and each action is the column where a piece was placed.
    
    :param params: Tuple containing (search_time, random_seed)
    :return: List of [state + action] pairs
    """
    search_time, seed = params
    np.random.seed(seed)
    
    # Initialize empty board (6x7)
    board = np.zeros((s.ROWS, s.COLUMNS), dtype=int)
    records: List[List[int]] = []
    
    # Players alternate turns (1: human, 2: AI)
    turn = s.FIRST_PLAYER_PIECE  # Start with player 1
    
    # While the game is not over
    while True:
        # Get board state before move (flatten to 1D)
        state = board_to_feature_vector(board)
        
        if turn == s.FIRST_PLAYER_PIECE:
            # For player 1, we'll use Monte Carlo for both players to simulate
            root = Node(board=board.copy(), last_player=s.SECOND_PLAYER_PIECE)
            mc = MonteCarlo(root)
            move = mc.start(search_time // 2)  # Half search time for faster simulation
        else:
            # For player 2 (AI), use Monte Carlo with full search time
            root = Node(board=board.copy(), last_player=s.FIRST_PLAYER_PIECE)
            mc = MonteCarlo(root)
            move = mc.start(search_time)
        
        # Record the state and the chosen move
        records.append(state + [move])
        
        # Make the move
        row = game.get_next_open_row(board, move)
        game.drop_piece(board, row, move, turn)
        
        # Check if game is over
        if game.winning_move(board, turn) or game.is_game_tied(board):
            break
            
        # Switch players
        turn = s.SECOND_PLAYER_PIECE if turn == s.FIRST_PLAYER_PIECE else s.FIRST_PLAYER_PIECE
    
    return records

def generate_dataset(
    n_games: int = 1000,
    search_time: int = 1,  # MCTS search time in seconds per move
    out_file: str = "connect4_dataset.csv"
) -> None:
    """
    Generates `n_games` Connect-Four games in parallel using MCTS,
    recording each pair (state, move) in CSV.
    
    :param n_games: Number of games to simulate
    :param search_time: Time in seconds for MCTS to search for each move
    :param out_file: Output CSV file name
    """
    path = os.path.join(DATA_DIR, out_file)
    
    # Write header
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        # 42 cells (6x7 board) plus the move column
        header = [f"cell_{i}" for i in range(s.ROWS * s.COLUMNS)] + ["move"]
        writer.writerow(header)
    
    # Determine number of processes (minimum of available CPUs and 8)
    n_processes = min(cpu_count(), 8)
    pool = Pool(processes=n_processes)
    print(f"Starting parallel generation with {n_processes} processes...")
    
    # Prepare tasks: each tuple is (search_time, random_seed)
    # Using different seeds for randomness
    tasks = [(search_time, i) for i in range(n_games)]
    
    # Track progress
    completed = 0
    start_time = time.time()
    
    # Process games in batches
    batch_size = min(100, max(1, n_games // 20))
    
    # Use imap to process games in parallel
    for batch_results in pool.imap_unordered(generate_game, tasks, chunksize=batch_size):
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(batch_results)
        
        completed += 1
        if completed % batch_size == 0 or completed == n_games:
            elapsed = time.time() - start_time
            est_total = (elapsed / completed) * n_games if completed > 0 else 0
            remaining = max(0, est_total - elapsed)
            print(f"[{time.strftime('%H:%M:%S')}] Completed {completed}/{n_games} games "
                  f"({completed/n_games*100:.1f}%) - ETA: {remaining/60:.1f} min")
    
    pool.close()
    pool.join()
    print(f"Dataset saved to: {path}")
    print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    # Example usage: 500 games with 1 second search time per move
    # Reduced from 1000 to 500 as MCTS is computationally expensive
    generate_dataset(n_games=500, search_time=1)