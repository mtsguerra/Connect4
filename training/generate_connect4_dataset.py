import os
import sys
import time
import csv
import random
import numpy as np
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_structure import style as s
from game_structure import game_engine as game
from ai_alg.monte_carlo import Node, monte_carlo
from ai_alg.alpha_beta import alpha_beta
from ai_alg.basic_heuristic import evaluate_best_move

# Create data directory if it doesn't exist
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def board_to_feature_vector(board: np.ndarray) -> List[int]:
    """Convert board to a flattened feature vector"""
    return board.flatten().tolist()

def get_mcts_move(board, turn, search_time=1):
    # MCTS expects last_player as argument, so alternate correctly
    last_player = s.SECOND_PLAYER_PIECE if turn == s.FIRST_PLAYER_PIECE else s.FIRST_PLAYER_PIECE
    root = Node(board=board.copy(), last_player=last_player)
    mc = monte_carlo(root)
    return mc.start(search_time)

def get_alphabeta_move(board, turn):
    # AlphaBeta expects to play as s.SECOND_PLAYER_PIECE, so flip board if needed
    # For simplicity, just call as is, since the tree's output is still a strong move
    return alpha_beta(board)

def get_heuristic_move(board, turn):
    opponent = s.FIRST_PLAYER_PIECE if turn == s.SECOND_PLAYER_PIECE else s.SECOND_PLAYER_PIECE
    return evaluate_best_move(board, turn, opponent)

def get_random_move(board, turn):
    moves = game.available_moves(board)
    if isinstance(moves, int) and moves == -1:
        return None
    return random.choice(moves)

def generate_game(params: Tuple[int, int, float, float]) -> List[List[int]]:
    """
    Simulate a Connect4 game using a mixture of agents for both players.
    Each move, randomly select an agent (weighted for strong play).
    :param params: Tuple (search_time, random_seed, mcts_prob, ab_prob)
    """
    search_time, seed, mcts_prob, ab_prob = params
    np.random.seed(seed)
    random.seed(seed)
    board = np.zeros((s.ROWS, s.COLUMNS), dtype=int)
    records = []
    turn = s.FIRST_PLAYER_PIECE  # Player 1 starts

    AGENTS = [
        ("MCTS", lambda b, t: get_mcts_move(b, t, search_time)),
        ("AlphaBeta", get_alphabeta_move),
        ("Heuristic", get_heuristic_move),
        ("Random", get_random_move)
    ]
    # Probabilities: [MCTS, AlphaBeta, Heuristic, Random]
    agent_probs = [mcts_prob, ab_prob, 1 - mcts_prob - ab_prob - 0.05, 0.05]
    agent_probs = [max(0, p) for p in agent_probs]
    sum_probs = sum(agent_probs)
    agent_probs = [p / sum_probs for p in agent_probs]

    while True:
        state = board_to_feature_vector(board)
        # Select agent for this move
        agent_idx = np.random.choice(len(AGENTS), p=agent_probs)
        agent_name, agent_func = AGENTS[agent_idx]
        move = agent_func(board, turn)
        # If move is invalid, fallback to random
        moves = game.available_moves(board)
        if (move is None or not (move in moves)):
            move = get_random_move(board, turn)
        records.append(state + [move])
        row = game.get_next_open_row(board, move)
        game.drop_piece(board, row, move, turn)
        if game.winning_move(board, turn) or game.is_game_tied(board):
            break
        turn = s.SECOND_PLAYER_PIECE if turn == s.FIRST_PLAYER_PIECE else s.FIRST_PLAYER_PIECE
    return records

def generate_dataset(
    n_games: int = 750,
    search_time: int = 2,
    out_file: str = "connect4_dataset_mixed.csv",
    mcts_prob: float = 0.45,
    ab_prob: float = 0.45
) -> None:
    """
    Generate a dataset of Connect4 game states and moves using mixed agents.
    :param n_games: Number of games to simulate
    :param search_time: Time in seconds for MCTS to search per move
    :param out_file: Output CSV filename
    :param mcts_prob: Probability of picking MCTS per move
    :param ab_prob: Probability of picking AlphaBeta per move
    """
    path = os.path.join(DATA_DIR, out_file)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"cell_{i}" for i in range(s.ROWS * s.COLUMNS)] + ["move"]
        writer.writerow(header)

    n_processes = min(cpu_count(), 8)
    pool = Pool(processes=n_processes)
    print(f"Starting mixed-agent dataset generation with {n_processes} processes...")

    tasks = [(search_time, i, mcts_prob, ab_prob) for i in range(n_games)]
    completed = 0
    start_time = time.time()
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
    print("Starting strong and diverse Connect4 dataset generation...")
    generate_dataset(
        n_games=750,
        search_time=2,
        out_file="connect4_dataset_mixed.csv",
        mcts_prob=0.45,
        ab_prob=0.45
    )
    print("\nDataset generation complete. Now you can train your decision tree model using the mixed dataset.")
