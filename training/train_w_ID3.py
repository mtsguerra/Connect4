import pandas as pd
import numpy as np
import os
import sys
import pickle
from datetime import datetime
import random

# Add the root directory to the path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your ID3 implementation
from ai_alg.ID3 import ID3Tree

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

def calculate_confusion_matrix(y_true, y_pred):
    """
    Calculate confusion matrix without scikit-learn.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
        
    Returns:
    --------
    numpy.ndarray
        Confusion matrix where rows are true labels and columns are predicted labels.
    """
    labels = sorted(set(np.concatenate([y_true, y_pred])))
    n_labels = len(labels)
    
    # Create a label-to-index mapping
    label_to_index = {label: i for i, label in enumerate(labels)}
    
    # Initialize confusion matrix
    conf_matrix = np.zeros((n_labels, n_labels), dtype=int)
    
    # Fill confusion matrix
    for true_val, pred_val in zip(y_true, y_pred):
        conf_matrix[label_to_index[true_val], label_to_index[pred_val]] += 1
        
    return conf_matrix, labels

def train_id3_model(dataset_path=None, max_depth=20, train_fraction=0.8):
    """
    Train an ID3 decision tree model on Connect4 dataset.
    
    Args:
        dataset_path (str): Path to the dataset CSV file
        max_depth (int): Maximum depth of the decision tree
        train_fraction (float): Fraction of data to use for training
    
    Returns:
        tuple: Trained model and accuracy on test set
    """
    # Set default dataset path if not provided
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   "data", "connect4_dataset.csv")
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Check if dataset file exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Load the dataset
    data = pd.read_csv(dataset_path)
    print(f"Dataset shape: {data.shape}")
    print(data.head())
    
    # Separate features (X) and target (y)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Manually split into training and testing sets
    X_train, X_test, y_train, y_test = manual_train_test_split(
        X, y, test_size=(1-train_fraction), random_seed=42)
    
    # Convert to DataFrame and Series as required by your ID3 implementation
    X_train = pd.DataFrame(X_train, columns=[f'cell_{i}' for i in range(42)])
    X_test = pd.DataFrame(X_test, columns=[f'cell_{i}' for i in range(42)])
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    
    print(f'Training set size: {len(X_train)}')
    print(f'Testing set size: {len(X_test)}')
    
    # Train the ID3 tree
    print(f"Training ID3 tree with max_depth={max_depth}...")
    start_time = datetime.now()
    id3_tree = ID3Tree(max_depth=max_depth)
    id3_tree.fit(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Make predictions on test set
    print("Evaluating model on test set...")
    y_pred = id3_tree.predict(X_test)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test) * 100
    print(f'ID3 Tree Accuracy: {accuracy:.2f}%')
    
    # Calculate confusion matrix
    conf_matrix, labels = calculate_confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Create directory for models if it doesn't exist
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             "ai_alg", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the trained model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"id3_model_depth{max_depth}_{timestamp}.pkl"
    model_path = os.path.join(models_dir, model_filename)
    
    with open(model_path, "wb") as file:
        pickle.dump(id3_tree, file)
    print(f"ID3 model saved to: {model_path}")
    
    # Also save a standard named model for easy loading
    standard_model_path = os.path.join(models_dir, "id3_model.pkl")
    with open(standard_model_path, "wb") as file:
        pickle.dump(id3_tree, file)
    print(f"Standard model saved to: {standard_model_path}")
    
    # Return the trained model and accuracy
    return id3_tree, accuracy

def visualize_confusion_matrix(conf_matrix, labels):
    """
    Visualize confusion matrix without using scikit-learn.
    
    Parameters:
    -----------
    conf_matrix : numpy.ndarray
        Confusion matrix.
    labels : list
        List of class labels.
    """
    import matplotlib.pyplot as plt
    
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
    
    # Save the figure
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              "data")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train ID3 model for Connect4')
    parser.add_argument('--dataset', type=str, help='Path to dataset CSV file')
    parser.add_argument('--depth', type=int, default=20, help='Maximum depth of the decision tree')
    parser.add_argument('--train-fraction', type=float, default=0.8, 
                        help='Fraction of data to use for training (0.0-1.0)')
    args = parser.parse_args()
    
    # Train the model
    model, accuracy = train_id3_model(
        dataset_path=args.dataset,
        max_depth=args.depth,
        train_fraction=args.train_fraction
    )