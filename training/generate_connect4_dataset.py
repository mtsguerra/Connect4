import os
import sys
import time
import csv
import random
import numpy as np
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_structure import style as s
from game_structure import game_engine as game
from ai_alg.monte_carlo import Node, monte_carlo
from ai_alg.alpha_beta import alpha_beta
from ai_alg.basic_heuristic import evaluate_best_move

# Cria diretório de dados se não existir
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def board_to_feature_vector(board: np.ndarray) -> List[int]:
    """Converte o tabuleiro em um vetor de características flatten"""
    
    return board.flatten().tolist()

def get_mcts_move(board, turn, search_time=1):
    """retorna o movimento do mcts após esperar o último jogador"""
    
    last_player = s.SECOND_PLAYER_PIECE if turn == s.FIRST_PLAYER_PIECE else s.FIRST_PLAYER_PIECE
    root = Node(board=board.copy(), last_player=last_player)
    mc = monte_carlo(root)
    return mc.start(search_time)

def get_alphabeta_move(board, turn):
    """retorna o movimento realizado pelo alpha-beta"""
    
    return alpha_beta(board)

def get_heuristic_move(board, turn):
    """retorna o movimento realizado pelo A*"""
    
    opponent = s.FIRST_PLAYER_PIECE if turn == s.SECOND_PLAYER_PIECE else s.SECOND_PLAYER_PIECE
    return evaluate_best_move(board, turn, opponent)

def get_random_move(board, turn):
    """retorna um movimento aleatório"""
    
    moves = game.available_moves(board)
    if isinstance(moves, int) and moves == -1:
        return None
    return random.choice(moves)

def generate_game(params: Tuple[int, int, float, float]) -> List[List[int]]:
    """Simula partidas utilizando diferentes IAs, misturando-as e selecionando aleatoriamente para focar em aumentar a diversidade e qualidade do dataset formado"""
    
    search_time, seed, mcts_prob, ab_prob = params
    np.random.seed(seed)
    random.seed(seed)
    board = np.zeros((s.ROWS, s.COLUMNS), dtype=int)
    records = []
    turn = s.FIRST_PLAYER_PIECE

    AGENTS = [
        ("MCTS", lambda b, t: get_mcts_move(b, t, search_time)),
        ("AlphaBeta", get_alphabeta_move),
        ("Heurística", get_heuristic_move),
        ("Aleatório", get_random_move)
    ]
    # Probabilidades: [MCTS, AlphaBeta, Heurística, Aleatório]
    agent_probs = [mcts_prob, ab_prob, 1 - mcts_prob - ab_prob - 0.05, 0.05]
    agent_probs = [max(0, p) for p in agent_probs]
    sum_probs = sum(agent_probs)
    agent_probs = [p / sum_probs for p in agent_probs]

    while True:
        state = board_to_feature_vector(board)
        agent_idx = np.random.choice(len(AGENTS), p=agent_probs)
        agent_name, agent_func = AGENTS[agent_idx]
        move = agent_func(board, turn)
        # Se a jogada for inválida, tenta aleatório
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

def generate_dataset(n_games: int = 750, search_time: int = 2, out_file: str = "connect4_dataset_mixed.csv", mcts_prob: float = 0.45, ab_prob: float = 0.4) -> None:
    """Gera o dataset dos estados dos jogos e jogadas realizadas pelas múltiplas escolhas realizadas entre as IAs"""
    
    path = os.path.join(DATA_DIR, out_file)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"cell_{i}" for i in range(s.ROWS * s.COLUMNS)] + ["move"]
        writer.writerow(header)

    n_processes = min(cpu_count(), 8)
    pool = Pool(processes=n_processes)
    print(f"Iniciando geração de dataset misto com {n_processes} processos...")

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
            print(f"[{time.strftime('%H:%M:%S')}] {completed}/{n_games} partidas geradas "
                  f"({completed/n_games*100:.1f}%) - ETA: {eta/60:.1f} min")
    pool.close()
    pool.join()
    print(f"Dataset salvo em: {path}")
    print(f"Tempo total: {(time.time() - start_time)/60:.2f} minutos")

if __name__ == "__main__":
    print("Iniciando a geração do dataset forte e diverso de Connect4...")
    generate_dataset(
        n_games=750,
        search_time=2,
        out_file="connect4_dataset_mixed.csv",
        mcts_prob=0.45,
        ab_prob=0.45
    )
    print("\nGeração do dataset finalizada.")
