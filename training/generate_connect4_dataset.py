import os
import sys
import time
import csv
import numpy as np
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_structure import style as s
from game_structure import game_engine as game
from ai_alg.monte_carlo import Node, monte_carlo

# Create data directory if it doesn't exist
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def board_to_feature_vector(board: np.ndarray) -> List[int]:
    """Convert board to a flattened feature vector"""
    return board.flatten().tolist()

def generate_game(params: Tuple[int, int]) -> List[List[int]]:
    """
    Simulates a Connect4 game using Monte Carlo Tree Search.
    Returns a list of state-action pairs, where each state is a flattened board
    and each action is the column chosen.
    
    :param params: Tuple containing (search_time, random_seed)
    :return: List of [state + action] pairs
    """
    search_time, seed = params
    np.random.seed(seed)
    
    # Initialize empty board
    board = np.zeros((s.ROWS, s.COLUMNS), dtype=int)
    records = []
    
    # Track current player
    turn = s.FIRST_PLAYER_PIECE  # Start with player 1
    
    # Play until game over
    while True:
        # Record current state
        state = board_to_feature_vector(board)
        
        # Get move from MCTS
        root = Node(board=board.copy(), last_player=s.SECOND_PLAYER_PIECE if turn == s.FIRST_PLAYER_PIECE else s.FIRST_PLAYER_PIECE)
        mc = monte_carlo(root)
        move = mc.start(search_time)
        
        # Record state and chosen move
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
    n_games: int = 100,    # Generate fewer games as MCTS is computationally intensive
    search_time: int = 1,   # 1 second per move
    out_file: str = "connect4_dataset.csv"
) -> None:
    """
    Generate a dataset of Connect4 game states and moves using MCTS.
    
    :param n_games: Number of games to simulate
    :param search_time: Time in seconds for MCTS to search per move
    :param out_file: Output CSV filename
    """
    path = os.path.join(DATA_DIR, out_file)
    
    # Write header
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        # 42 cells (6x7 board) plus move column
        header = [f"cell_{i}" for i in range(s.ROWS * s.COLUMNS)] + ["move"]
        writer.writerow(header)
    
    # Use multiprocessing for parallel generation
    n_processes = min(cpu_count(), 8)  # Limit to prevent system overload
    pool = Pool(processes=n_processes)
    print(f"Starting dataset generation with {n_processes} processes...")
    
    # Prepare tasks (search_time, random_seed)
    tasks = [(search_time, i) for i in range(n_games)]
    
    # Track progress
    completed = 0
    start_time = time.time()
    
    # Process in batches
    batch_size = min(10, max(1, n_games // 10))
    
    for batch_results in pool.imap_unordered(generate_game, tasks, chunksize=batch_size):
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(batch_results)
        
        completed += 1
        if completed % batch_size == 0 or completed == n_games:
            elapsed = time.time() - start_time
            eta = (elapsed / completed) * (n_games - completed) if completed > 0 else 0
            print(f"[{time.strftime('%H:%M:%S')}] Completed {completed}/{n_games} games "
                  f"({completed/n_games*100:.1f}%) - ETA: {eta/60:.1f} min")
    
    pool.close()
    pool.join()
    print(f"Dataset saved to: {path}")
    print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Connect4 dataset using MCTS")
    parser.add_argument("--games", type=int, default=100, help="Number of games to generate")
    parser.add_argument("--time", type=float, default=1.0, help="Search time per move (seconds)")
    parser.add_argument("--output", type=str, default="connect4_dataset.csv", help="Output filename")
    args = parser.parse_args()
    
    generate_dataset(
        n_games=args.games,
        search_time=args.time,
        out_file=args.output
    )
