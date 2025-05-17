import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from collections import Counter

def load_dataset(dataset_path):
    """Load the Connect4 dataset from CSV file"""
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Number of samples: {len(df)}")
    return df

def analyze_move_distribution(df):
    """Analyze the distribution of moves in the dataset"""
    # Count the occurrences of each move
    move_counts = Counter(df['move'])
    total_moves = len(df)
    
    # Calculate percentages
    move_percentages = {move: count/total_moves*100 for move, count in move_counts.items()}
    
    print("\nMove Distribution:")
    for move in sorted(move_counts.keys()):
        print(f"Column {move}: {move_counts[move]} moves ({move_percentages[move]:.2f}%)")
    
    # Visualize move distribution
    plt.figure(figsize=(10, 6))
    moves = sorted(move_counts.keys())
    counts = [move_counts[move] for move in moves]
    
    plt.bar(moves, counts)
    plt.title("Distribution of Moves in Dataset")
    plt.xlabel("Column")
    plt.ylabel("Count")
    plt.xticks(moves)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "move_distribution.png"))
    plt.close()
    
    return move_counts

def analyze_board_states(df):
    """Analyze the board states in the dataset"""
    # Select only the board state columns
    board_states = df.iloc[:, :-1]
    
    # Count empty spaces, player 1 pieces, and player 2 pieces
    empty_count = (board_states == 0).sum().sum()
    p1_count = (board_states == 1).sum().sum()
    p2_count = (board_states == 2).sum().sum()
    
    total_cells = board_states.shape[0] * board_states.shape[1]
    
    print("\nBoard State Analysis:")
    print(f"Empty cells: {empty_count} ({empty_count/total_cells*100:.2f}%)")
    print(f"Player 1 pieces: {p1_count} ({p1_count/total_cells*100:.2f}%)")
    print(f"Player 2 pieces: {p2_count} ({p2_count/total_cells*100:.2f}%)")
    
    # Calculate average number of pieces per board
    avg_pieces = (p1_count + p2_count) / len(board_states)
    print(f"Average number of pieces per board: {avg_pieces:.2f}")
    
    return {
        'empty': empty_count,
        'player1': p1_count,
        'player2': p2_count,
        'avg_pieces': avg_pieces
    }

def manual_train_test_split(X, y, test_size=0.2, random_seed=42):
    """
    Split arrays into random train and test subsets without using scikit-learn.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Features data.
    y : array-like, shape (n_samples,)
        Target data.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_seed : int, default=42
        Random seed for reproducibility.
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        The split data.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size should be between 0 and 1")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    n_samples = len(X)
    indices = list(range(n_samples))
    
    # Shuffle indices
    random.shuffle(indices)
    
    # Calculate split point
    test_samples = int(n_samples * test_size)
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]
    
    # Split the data
    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_indices].copy()
        X_test = X.iloc[test_indices].copy()
    else:
        X_train = X[train_indices].copy()
        X_test = X[test_indices].copy()
        
    if isinstance(y, pd.Series):
        y_train = y.iloc[train_indices].copy()
        y_test = y.iloc[test_indices].copy()
    else:
        y_train = y[train_indices].copy()
        y_test = y[test_indices].copy()
    
    return X_train, X_test, y_train, y_test

def evaluate_model(y_true, y_pred):
    """
    Calculate evaluation metrics for model performance without using scikit-learn.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics.
    """
    # Convert to numpy arrays if they aren't already
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    
    # Calculate accuracy
    accuracy = np.mean(y_true == y_pred) * 100
    
    # Calculate confusion matrix
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    n_labels = len(unique_labels)
    conf_matrix = np.zeros((n_labels, n_labels), dtype=int)
    
    for i, true_label in enumerate(unique_labels):
        for j, pred_label in enumerate(unique_labels):
            conf_matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    
    # Calculate precision, recall, and F1 score for each class
    precision = {}
    recall = {}
    f1_score = {}
    
    for i, label in enumerate(unique_labels):
        # True positives
        tp = conf_matrix[i, i]
        # Sum of row (all actual positives)
        actual_positives = np.sum(conf_matrix[i, :])
        # Sum of column (all predicted positives)
        predicted_positives = np.sum(conf_matrix[:, i])
        
        # Calculate metrics, handling division by zero
        prec = tp / predicted_positives if predicted_positives > 0 else 0
        rec = tp / actual_positives if actual_positives > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        precision[label] = prec
        recall[label] = rec
        f1_score[label] = f1
    
    # Calculate macro-averaged metrics
    macro_precision = np.mean(list(precision.values()))
    macro_recall = np.mean(list(recall.values()))
    macro_f1 = np.mean(list(f1_score.values()))
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'labels': unique_labels,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }

def plot_confusion_matrix(conf_matrix, labels, output_path=None):
    """Plot confusion matrix without using scikit-learn."""
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    
    # Set x and y ticks
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    
    # Add text annotations
    thresh = conf_matrix.max() / 2.0
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Column')
    plt.xlabel('Predicted Column')
    
    if output_path:
        plt.savefig(output_path)
    plt.close()

def analyze_dataset(dataset_path=None):
    """
    Analyze the Connect4 dataset and evaluate a simple baseline model.
    
    Parameters:
    -----------
    dataset_path : str, optional
        Path to the CSV dataset file.
    """
    # Set default dataset path if not provided
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "connect4_dataset.csv")
    
    # Load the dataset
    df = load_dataset(dataset_path)
    
    # Analyze move distribution
    move_counts = analyze_move_distribution(df)
    
    # Analyze board states
    state_stats = analyze_board_states(df)
    
    # Split features and target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Manually split the data
    X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_size=0.2)
    
    print(f"\nSplit Sizes:")
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    # Create a simple baseline model (most common move from training set)
    most_common_move = Counter(y_train).most_common(1)[0][0]
    print(f"\nBaseline Model: Always predicts column {most_common_move}")
    
    # Make predictions with the baseline
    y_pred = np.full_like(y_test, most_common_move)
    
    # Evaluate the baseline model
    eval_metrics = evaluate_model(y_test, y_pred)
    
    print(f"\nBaseline Model Evaluation:")
    print(f"Accuracy: {eval_metrics['accuracy']:.2f}%")
    print(f"Macro-averaged Precision: {eval_metrics['macro_precision']:.4f}")
    print(f"Macro-averaged Recall: {eval_metrics['macro_recall']:.4f}")
    print(f"Macro-averaged F1 Score: {eval_metrics['macro_f1']:.4f}")
    
    # Plot confusion matrix
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)
    plot_confusion_matrix(
        eval_metrics['confusion_matrix'],
        eval_metrics['labels'],
        os.path.join(output_dir, "baseline_confusion_matrix.png")
    )
    
    return {
        'dataset_stats': {
            'n_samples': len(df),
            'move_distribution': move_counts,
            'board_state_stats': state_stats
        },
        'baseline_metrics': eval_metrics
    }

def visualize_board_position_frequencies(df):
    """
    Visualize the frequency of pieces in each board position.
    
    Parameters:
    -----------
    df : DataFrame
        The Connect4 dataset.
    """
    # Get board state columns (all except the last column)
    board_states = df.iloc[:, :-1]
    
    # Count occurrences of each player's pieces in each position
    p1_freq = (board_states == 1).sum() / len(board_states)
    p2_freq = (board_states == 2).sum() / len(board_states)
    
    # Reshape to 6x7 grid (Connect4 board dimensions)
    p1_grid = np.array(p1_freq).reshape(6, 7)
    p2_grid = np.array(p2_freq).reshape(6, 7)
    
    # Plot heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Player 1 heatmap
    im1 = ax1.imshow(p1_grid, cmap='Blues')
    ax1.set_title("Player 1 Piece Frequency")
    ax1.set_xlabel("Column")
    ax1.set_ylabel("Row")
    ax1.set_xticks(np.arange(7))
    ax1.set_yticks(np.arange(6))
    fig.colorbar(im1, ax=ax1, label='Frequency')
    
    # Add text annotations
    for i in range(6):
        for j in range(7):
            text = ax1.text(j, i, f"{p1_grid[i, j]:.2f}",
                           ha="center", va="center", color="black" if p1_grid[i, j] < 0.5 else "white")
    
    # Player 2 heatmap
    im2 = ax2.imshow(p2_grid, cmap='Reds')
    ax2.set_title("Player 2 Piece Frequency")
    ax2.set_xlabel("Column")
    ax2.set_ylabel("Row")
    ax2.set_xticks(np.arange(7))
    ax2.set_yticks(np.arange(6))
    fig.colorbar(im2, ax=ax2, label='Frequency')
    
    # Add text annotations
    for i in range(6):
        for j in range(7):
            text = ax2.text(j, i, f"{p2_grid[i, j]:.2f}",
                           ha="center", va="center", color="black" if p2_grid[i, j] < 0.5 else "white")
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "piece_frequency_heatmaps.png"))
    plt.close()

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Analyze Connect4 dataset")
    parser.add_argument('--dataset', type=str, help='Path to dataset CSV file')
    args = parser.parse_args()
    
    # Analyze the dataset
    stats = analyze_dataset(dataset_path=args.dataset)
    
    # If dataset was successfully loaded, visualize board position frequencies
    if args.dataset:
        df = pd.read_csv(args.dataset)
        visualize_board_position_frequencies(df)
