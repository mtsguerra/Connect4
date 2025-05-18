import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter

def load_dataset(dataset_path):
    """Carrega o dataset a partir do arquivo CSV"""
    
    print(f"Carregando dataset de {dataset_path}...")
    df = pd.read_csv(dataset_path)
    print(f"Shape do dataset: {df.shape}")
    print(f"Número de amostras: {len(df)}")
    return df

def analyze_move_distribution(df):
    """Analisa e organiza a distribuição dos movimentos no dataset"""
    
    # Conta a ocorrência de cada movimento
    move_counts = Counter(df['move'])
    total_moves = len(df)
    
    # Calcula a porcentagem de cada movimento
    move_percentages = {move: count/total_moves*100 for move, count in move_counts.items()}
    
    print("\nDistribuição dos Movimentos:")
    for move in sorted(move_counts.keys()):
        print(f"Coluna {move}: {move_counts[move]} movimentos ({move_percentages[move]:.2f}%)")
    
    # Cria o gráfico de barras da distribuição dos movimentos
    plt.figure(figsize=(10, 6))
    moves = sorted(move_counts.keys())
    counts = [move_counts[move] for move in moves]
    
    plt.bar(moves, counts)
    plt.title("Distribution of Moves in Dataset")
    plt.xlabel("Column")
    plt.ylabel("Count")
    plt.xticks(moves)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Salva o gráfico na pasta data
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "move_distribution.png"))
    plt.close()
    print(f"Gráfico salvo em: {os.path.join(output_dir, 'move_distribution.png')}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analisar distribuição dos movimentos no Connect4")
    parser.add_argument('--dataset', type=str, help='Caminho para o arquivo CSV do dataset')
    args = parser.parse_args()
    
    if not args.dataset or not os.path.exists(args.dataset):
        print("Por favor, forneça um caminho válido para o dataset usando --dataset")
    else:
        df = load_dataset(args.dataset)
        analyze_move_distribution(df)
