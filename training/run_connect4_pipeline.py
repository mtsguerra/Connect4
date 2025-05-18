import os
import sys
import time

# Adiciona o diretório raiz do projeto ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_full_pipeline():
    """Executa o pipeline completo do Connect4: treina o modelo, testa o modelo"""
    
    start_time = time.time()
    
    # Como o dataset ja foi gerado em outro programa este é apenas para treinar
    
    print("\n=== ETAPA 1: Treinar Modelo de Árvore de Decisão ID3 ===")
    from decision_tree import train_iris_model, train_connect4_model
    
    # Treina no dataset iris como aquecimento
    print("\nTreinando no dataset iris (aquecimento):")
    train_iris_model(iris_path="Connect4-main/training/iris.csv")
    
    # Treina no dataset Connect4
    print("\nTreinando no dataset Connect4:")
    connect4_model = train_connect4_model(
        data_path="Connect4-main/training/data/connect4_dataset_mixed.csv",
        max_depth=10,
        save_path="Connect4-main/training/data/connect4_tree_model_mixed.pkl"
    )
    
    print("\n=== ETAPA 2: Comparar Jogadores MCTS e Árvore de Decisão ===")
    print("\nNota: Para comparar os jogadores MCTS e Árvore de Decisão, execute:")
    print("python Connect4-main/training/compare_ai_players.py --games 10")
    
    total_time = time.time() - start_time
    print(f"\nPipeline completo executado em {total_time/60:.2f} minutos")

if __name__ == "__main__":
    run_full_pipeline()
