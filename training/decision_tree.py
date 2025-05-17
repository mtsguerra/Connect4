import numpy as np
import pandas as pd
import os
import sys
import time
import pickle
from collections import Counter
from math import log

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Node:
    """Simple decision tree node"""
    def __init__(self):
        self.feature = None      # Feature to split on
        self.value = None        # Value of feature
        self.children = {}       # Child nodes
        self.is_leaf = False     # Is this a leaf node?
        self.prediction = None   # For leaf nodes, predicted class
        self.depth = 0           # Depth in the tree

class ID3Tree:
    """
    Simplified ID3 Decision Tree implementation for Connect4
    Custom implementation without using scikit-learn
    """
    def __init__(self, max_depth=10):
        """Initialize the tree with configurable max depth"""
        self.root = None
        self.max_depth = max_depth
    
    def fit(self, X, y):
        """Build the decision tree from training data"""
        print("Training ID3 decision tree...")
        start_time = time.time()
        
        # Convert to numpy arrays if needed
        X = np.array(X)
        y = np.array(y)
        
        # Get feature indices (0 to n_features-1)
        features = list(range(X.shape[1]))
        
        # Build the tree recursively
        self.root = self._build_tree(X, y, features, depth=0)
        
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
        return self
    
    def predict(self, X):
        """Predict classes for samples in X"""
        X = np.array(X)
        return np.array([self._predict_sample(x, self.root) for x in X])
    
    def _predict_sample(self, x, node):
        """Predict class for a single sample"""
        # If leaf node, return prediction
        if node.is_leaf:
            return node.prediction
        
        # Get the value for the feature we're splitting on
        feature_value = x[node.feature]
        
        # If we haven't seen this value during training, use most common child value
        if feature_value not in node.children:
            # Find most common prediction in child nodes
            predictions = [child.prediction for child in node.children.values() if child.is_leaf]
            if not predictions:
                # If no leaf children, just pick a random child
                return list(node.children.values())[0].prediction
            else:
                return Counter(predictions).most_common(1)[0][0]
        
        # Recurse down the tree
        return self._predict_sample(x, node.children[feature_value])
    
    def _build_tree(self, X, y, features, depth):
        """Recursively build the decision tree"""
        node = Node()
        node.depth = depth
        
        # If all samples have the same class, create a leaf node
        if len(np.unique(y)) == 1:
            node.is_leaf = True
            node.prediction = y[0]
            return node
        
        # If no features left to split on or max depth reached, create a leaf node
        if len(features) == 0 or depth >= self.max_depth:
            node.is_leaf = True
            # Use most common class
            node.prediction = Counter(y).most_common(1)[0][0]
            return node
        
        # Find the best feature to split on (highest information gain)
        best_feature = self._find_best_feature(X, y, features)
        node.feature = best_feature
        
        # Get unique values for the best feature
        unique_values = np.unique(X[:, best_feature])
        
        # Create child nodes for each value
        new_features = features.copy()
        new_features.remove(best_feature)
        
        # Check if there are enough samples to split on
        if len(unique_values) <= 1:
            node.is_leaf = True
            node.prediction = Counter(y).most_common(1)[0][0]
            return node
        
        # Create child nodes
        for value in unique_values:
            # Get samples with this value
            indices = np.where(X[:, best_feature] == value)[0]
            
            # If no samples with this value, skip
            if len(indices) == 0:
                continue
            
            # Create child node
            child = self._build_tree(X[indices], y[indices], new_features, depth + 1)
            node.children[value] = child
            
        # If no child nodes created (shouldn't happen), create a leaf node
        if not node.children:
            node.is_leaf = True
            node.prediction = Counter(y).most_common(1)[0][0]
            
        return node
    
    def _find_best_feature(self, X, y, features):
        """Find the feature with the highest information gain"""
        best_gain = -1
        best_feature = None
        
        # Calculate base entropy
        base_entropy = self._entropy(y)
        
        for feature in features:
            # Skip features with no variance
            if len(np.unique(X[:, feature])) <= 1:
                continue
                
            # Calculate information gain
            gain = self._information_gain(X, y, feature, base_entropy)
            
            # Update best feature if this gain is higher
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        
        # If no good feature found, just pick the first one
        if best_feature is None and features:
            best_feature = features[0]
            
        return best_feature
    
    def _entropy(self, y):
        """Calculate entropy of a class distribution"""
        counts = Counter(y)
        total = len(y)
        entropy = 0
        
        for count in counts.values():
            p = count / total
            entropy -= p * log(p, 2)
            
        return entropy
    
    def _information_gain(self, X, y, feature, base_entropy):
        """Calculate information gain for a feature"""
        values = X[:, feature]
        unique_values = np.unique(values)
        
        # Calculate weighted entropy
        weighted_entropy = 0
        total_samples = len(y)
        
        for value in unique_values:
            # Get samples with this value
            indices = np.where(values == value)[0]
            subset_y = y[indices]
            
            # Calculate weight and entropy
            weight = len(subset_y) / total_samples
            weighted_entropy += weight * self._entropy(subset_y)
        
        # Calculate information gain
        return base_entropy - weighted_entropy
    
    def save(self, filename):
        """Save the model to a file"""
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
            
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filename}")
    
    @staticmethod
    def load(filename):
        """Load model from file"""
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filename}")
        return model
    
    def print_tree(self, max_depth=3):
        """Print a text representation of the tree"""
        if self.root is None:
            print("Tree not built yet")
            return
        
        self._print_node(self.root, "", True, max_depth)
    
    def _print_node(self, node, indent, is_last, max_depth=None):
        """Print a single node and its children"""
        if max_depth is not None and node.depth > max_depth:
            return
            
        # Print the current node
        if node.is_leaf:
            print(f"{indent}{'└── ' if is_last else '├── '}Prediction: {node.prediction}")
        else:
            print(f"{indent}{'└── ' if is_last else '├── '}Feature: {node.feature}")
            
            # Print child nodes
            child_keys = list(node.children.keys())
            for i, value in enumerate(child_keys):
                child = node.children[value]
                is_last_child = (i == len(child_keys) - 1)
                
                # Add indentation
                new_indent = indent + ("    " if is_last else "│   ")
                print(f"{new_indent}{'└── ' if is_last_child else '├── '}Value: {value}")
                
                self._print_node(child, new_indent + "    ", is_last_child, max_depth)

def load_and_split_data(file_path, test_size=0.2, random_seed=42):
    """Load data from CSV and split into train/test sets"""
    print(f"Loading dataset from {file_path}...")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return None, None, None, None
    
    # Load data
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    
    # Split features and target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Generate indices for train/test split
    indices = np.random.permutation(len(X))
    test_size_int = int(test_size * len(X))
    test_indices = indices[:test_size_int]
    train_indices = indices[test_size_int:]
    
    # Split data
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    return X_train, y_train, X_test, y_test

def evaluate_model(y_true, y_pred):
    """Calculate and print model performance metrics"""
    # Calculate accuracy
    accuracy = np.mean(y_true == y_pred) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Calculate class distribution
    class_dist = Counter(y_pred)
    print("\nPredicted class distribution:")
    for col, count in sorted(class_dist.items()):
        print(f"Column {col}: {count} ({count/len(y_pred)*100:.2f}%)")
    
    # Create confusion matrix
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    conf_matrix = np.zeros((len(classes), len(classes)), dtype=int)
    
    for true_idx, true_val in enumerate(classes):
        for pred_idx, pred_val in enumerate(classes):
            conf_matrix[true_idx, pred_idx] = np.sum((y_true == true_val) & (y_pred == pred_val))
    
    print("\nConfusion Matrix:")
    # Print header
    header = " " * 10
    for c in classes:
        header += f"Pred {c:<5} "
    print(header)
    
    # Print rows
    for i, c in enumerate(classes):
        row = f"True {c:<5} "
        for j in range(len(classes)):
            row += f"{conf_matrix[i, j]:<10} "
        print(row)
    
    # Calculate per-class metrics
    for c in classes:
        true_positives = np.sum((y_true == c) & (y_pred == c))
        all_predicted = np.sum(y_pred == c)
        all_actual = np.sum(y_true == c)
        
        precision = true_positives / all_predicted if all_predicted > 0 else 0
        recall = true_positives / all_actual if all_actual > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nClass {c} metrics:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")
    
    return accuracy, conf_matrix

def train_connect4_model(data_path=None, max_depth=10, save_path=None):
    """Train a decision tree on Connect4 dataset"""
    # Set default paths if not provided
    if data_path is None:
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                "data", "connect4_dataset.csv")
    
    if save_path is None:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "data", "connect4_tree_model.pkl")
    
    # Load and split data
    X_train, y_train, X_test, y_test = load_and_split_data(data_path, test_size=0.2)
    
    if X_train is None:
        print("Error loading data. Exiting.")
        return None
    
    # Train model
    model = ID3Tree(max_depth=max_depth)
    model.fit(X_train, y_train)
    
    # Evaluate on training data
    print("\nEvaluating on training data:")
    y_train_pred = model.predict(X_train)
    train_acc, _ = evaluate_model(y_train, y_train_pred)
    
    # Evaluate on test data
    print("\nEvaluating on test data:")
    y_test_pred = model.predict(X_test)
    test_acc, _ = evaluate_model(y_test, y_test_pred)
    
    # Save model
    if save_path:
        model.save(save_path)
    
    # Print tree structure (limited to 3 levels for readability)
    print("\nTree Structure (limited to 3 levels):")
    model.print_tree(max_depth=3)
    
    return model

def train_iris_model(iris_path=None):
    """
    Train a decision tree on the iris dataset
    
    Parameters:
    -----------
    iris_path : str, optional
        Path to the iris dataset CSV file. If not provided,
        will use the default path in the repository.
    """
    # Set default path if not provided
    if iris_path is None:
        iris_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "iris.csv"
        )
        
    print(f"Loading iris dataset from {iris_path}...")
    
    # Check if file exists
    if not os.path.exists(iris_path):
        print(f"Error: Iris dataset not found at {iris_path}")
        return None
        
    # Load data
    try:
        df = pd.read_csv(iris_path)
        print(f"Loaded iris dataset with shape: {df.shape}")
        
        # Extract features (skip ID column if exists)
        feature_cols = [col for col in df.columns if col.lower() not in ['id', 'class']]
        X = df[feature_cols].values
        
        # Extract labels and convert to numeric
        # Map class names to integers: setosa->0, versicolor->1, virginica->2
        class_mapping = {
            'Iris-setosa': 0,
            'Iris-versicolor': 1,
            'Iris-virginica': 2
        }
        
        # Get class column
        class_col = [col for col in df.columns if 'class' in col.lower()][0]
        y = df[class_col].map(class_mapping).values
        
        # Check if mapping worked
        if np.any(pd.isna(y)):
            print("Warning: Some class labels could not be mapped.")
            print("Unique classes in the dataset:", df[class_col].unique())
            # Try to fix unmapped values
            unmapped = df.loc[pd.isna(df[class_col].map(class_mapping)), class_col].unique()
            for i, cls in enumerate(unmapped):
                print(f"Mapping '{cls}' to {i}")
                y[df[class_col] == cls] = i
        
        # Feature names
        feature_names = feature_cols
        print(f"Features: {feature_names}")
        print(f"Classes: {list(class_mapping.keys())}")
        
        # Split data - no scikit-learn needed
        np.random.seed(42)
        indices = np.random.permutation(len(X))
        train_indices = indices[:int(0.8 * len(X))]
        test_indices = indices[int(0.8 * len(X)):]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Discretize continuous features
        for i in range(X_train.shape[1]):
            # Create 3 bins based on percentiles
            values = X_train[:, i]
            thresholds = [
                np.percentile(values, 33),
                np.percentile(values, 66)
            ]
            
            # Apply discretization
            X_train[:, i] = np.digitize(X_train[:, i], thresholds)
            X_test[:, i] = np.digitize(X_test[:, i], thresholds)
        
        # Train model
        model = ID3Tree(max_depth=4)
        model.fit(X_train, y_train)
        
        # Evaluate
        print("\nIris Dataset Results:")
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_acc = np.mean(train_pred == y_train) * 100
        test_acc = np.mean(test_pred == y_test) * 100
        
        print(f"Training accuracy: {train_acc:.2f}%")
        print(f"Testing accuracy: {test_acc:.2f}%")
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        conf_matrix = np.zeros((3, 3), dtype=int)
        for i in range(len(test_pred)):
            conf_matrix[y_test[i]][test_pred[i]] += 1
        
        print("           Predicted")
        print("           0    1    2")
        print("Actual 0   {}   {}   {}".format(*conf_matrix[0]))
        print("       1   {}   {}   {}".format(*conf_matrix[1]))
        print("       2   {}   {}   {}".format(*conf_matrix[2]))
        
        # Print class names
        print("\nClass mapping:")
        for cls, idx in class_mapping.items():
            print(f"{idx}: {cls}")
        
        # Print tree
        print("\nIris Decision Tree:")
        model.print_tree()
        
        # Save model if data directory exists
        try:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
            os.makedirs(output_dir, exist_ok=True)
            model.save(os.path.join(output_dir, "iris_tree.pkl"))
        except:
            print("Could not save model")
        
        return model
        
    except Exception as e:
        print(f"Error loading iris dataset: {e}")
        import traceback
        traceback.print_exc()
        return None
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ID3 decision tree for Connect4")
    parser.add_argument("--data", type=str, help="Path to Connect4 dataset CSV")
    parser.add_argument("--iris-data", type=str, default="training/iris.csv", 
                        help="Path to iris dataset CSV")
    parser.add_argument("--depth", type=int, default=10, help="Maximum tree depth")
    parser.add_argument("--iris", action="store_true", help="Train on iris dataset first")
    parser.add_argument("--output", type=str, help="Path to save model")
    
    args = parser.parse_args()
    
    # Train on iris dataset if requested
    if args.iris:
        print("=== Training on Iris Dataset ===")
        train_iris_model(iris_path=args.iris_data)
        print("\n")
    
    # Train on Connect4 dataset
    print("=== Training on Connect4 Dataset ===")
    train_connect4_model(
        data_path=args.data,
        max_depth=args.depth,
        save_path=args.output
    )