import numpy as np
import csv
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime
from game_structure import style as s
from game_structure import game_engine as game
from ai_alg import basic_heuristic as b, alpha_beta as a, monte_carlo as m
from training.decision_tree_player import decision_tree_move

AI_TYPES = {
    "Easy": lambda board, player, opponent: b.evaluate_best_move(board, player, opponent),
    "Medium": lambda board, player, opponent: m.mcts(board),
    "Hard": lambda board, player, opponent: a.alpha_beta(board),
    "Challenge": lambda board, player, opponent: decision_tree_move(board),
}

AI_NAMES = list(AI_TYPES.keys())

def run_game(p1_ai, p2_ai, verbose=False):
    """Executa a partida entre as duas IAs e retorna suas estatísticas"""
    board = np.zeros((s.ROWS, s.COLUMNS), dtype=int)
    turn = 1
    move_count = 0
    history = []
    while True:
        current_ai = p1_ai if turn == 1 else p2_ai
        opponent = 1 if turn == 2 else 2
        move_func = AI_TYPES[current_ai]
        col = move_func(board.copy(), turn, opponent)
        if col is None:
            moves = game.available_moves(board)
            col = moves[0] if moves != -1 else 0
        row = game.get_next_open_row(board, col)
        if row == -1:
            moves = game.available_moves(board)
            col = moves[0] if moves != -1 else 0
            row = game.get_next_open_row(board, col)
        game.drop_piece(board, row, col, turn)
        move_count += 1
        history.append((turn, col))
        if verbose:
            print(f"Turn {move_count}: Player {turn} ({current_ai}) -> Col {col+1}")
        if game.winning_move(board, turn):
            winner = turn
            break
        if game.is_game_tied(board):
            winner = 0
            break
        turn = 2 if turn == 1 else 1
    return {
        "winner": winner,
        "moves": move_count,
        "history": history
    }

def main(num_matches_per_pair=10, csv_path="ai_vs_ai_results.csv"):
    """Executa partidas entre todos os diferentes pares de IAs e salva os resultados em um CSV"""
    results = []
    match_id = 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i, ai1 in enumerate(AI_NAMES):
        for ai2 in AI_NAMES[i+1:]:
            for match in range(num_matches_per_pair):
                # Alterna quem inicia cada partida, para melhor análise
                if match % 2 == 0:
                    p1, p2 = ai1, ai2
                else:
                    p1, p2 = ai2, ai1
                stats = run_game(p1, p2)
                winner_str = (
                    "Draw" if stats["winner"] == 0 else
                    ("Player1" if stats["winner"] == 1 else "Player2")
                )
                results.append({
                    "timestamp": now,
                    "match_id": match_id,
                    "player1_ai": p1,
                    "player2_ai": p2,
                    "winner": winner_str,
                    "total_moves": stats["moves"],
                    "moves_sequence": ";".join(f"P{turn}:{col+1}" for turn, col in stats["history"]),
                })
                print(f"Match {match_id}: {p1} (P1) vs {p2} (P2) => {winner_str}")
                match_id += 1
    fieldnames = ["timestamp", "match_id", "player1_ai", "player2_ai", "winner", "total_moves", "moves_sequence"]
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main(num_matches_per_pair=10)
