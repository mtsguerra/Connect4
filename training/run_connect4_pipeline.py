import os
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_full_pipeline():
    """Run the full Connect4 pipeline: generate data, train model, test model"""
    start_time = time.time()
    
    print("=== STEP 1: Generate Connect4 Dataset ===")
    from generate_connect4_dataset import generate_dataset
    
    # Generate dataset
    generate_dataset(
        n_games=200,
        search_time=1.0,
        out_file="connect4_dataset.csv"
    )
    
    print("\n=== STEP 2: Train ID3 Decision Tree Model ===")
    from training.decision_tree import train_iris_model, train_connect4_model
    
    # Train on iris dataset first as a warm-up
    print("\nTraining on iris dataset (warm-up):")
    train_iris_model(iris_path="training/iris.csv")
    
    # Train on Connect4 dataset
    print("\nTraining on Connect4 dataset:")
    connect4_model = train_connect4_model(
        data_path="training/data/connect4_dataset.csv",
        max_depth=10,
        save_path="training/data/connect4_tree_model.pkl"
    )
    
    print("\n=== STEP 3: Compare MCTS and Decision Tree Players ===")
    print("\nNote: To compare MCTS and Decision Tree players, run:")
    print("python training/compare_ai_players.py --games 10")
    
    total_time = time.time() - start_time
    print(f"\nFull pipeline completed in {total_time/60:.2f} minutes")

if __name__ == "__main__":
    run_full_pipeline()