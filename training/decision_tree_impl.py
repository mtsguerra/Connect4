import numpy as np
import math
import csv
from collections import Counter, defaultdict

class Node:
    """Decision Tree Node"""
    def __init__(self, attribute=None, value=None, result=None):
        self.attribute = attribute  # Which attribute this node tests
        self.value = value          # Value of the attribute from parent
        self.result = result        # Classification result (leaf nodes only)
        self.children = {}          # Dictionary of child nodes
        
    def __str__(self):
        if self.result is not None:
            return f"Result: {self.result}"
        return f"Attribute: {self.attribute}"


class DecisionTree:
    """
    Decision Tree implementation using ID3 algorithm
    """
    def __init__(self, discretize_numeric=True, bins=5):
        self.root = None
        self.attributes = []
        self.discretize_numeric = discretize_numeric
        self.bins = bins
        self.bin_thresholds = {}
        
    def fit(self, data, attributes, target_attr):
        """
        Build the decision tree using the ID3 algorithm
        
        Parameters:
        - data: List of dictionaries, each containing attribute-value pairs
        - attributes: List of attribute names
        - target_attr: Name of the target attribute to predict
        """
        # Store list of attributes for later use
        self.attributes = attributes.copy()
        
        # Discretize numeric attributes if enabled
        if self.discretize_numeric:
            data = self._discretize_data(data, attributes, target_attr)
            
        # Build the tree
        self.root = self._id3(data, attributes, target_attr)
        
    def _id3(self, data, attributes, target_attr):
        """
        ID3 algorithm implementation
        
        Parameters:
        - data: List of dictionaries containing attribute-value pairs
        - attributes: List of attributes to consider for splitting
        - target_attr: The target attribute to predict
        
        Returns:
        - A decision tree node
        """
        # Create a counter of all values for the target attribute
        target_counts = Counter([record[target_attr] for record in data])
        
        # If all records have the same classification, return a leaf node
        if len(target_counts) == 1:
            return Node(result=list(target_counts.keys())[0])
        
        # If there are no attributes left, return the most common value
        if not attributes:
            most_common = target_counts.most_common(1)[0][0]
            return Node(result=most_common)
        
        # Select the best attribute to split on
        best_attr = self._choose_attribute(data, attributes, target_attr)
        
        # Create a new decision tree node with the best attribute
        tree = Node(attribute=best_attr)
        
        # Create a new subtree for each value of the best attribute
        attr_values = set(record[best_attr] for record in data)
        
        for value in attr_values:
            # Create a subset of data where best_attr == value
            subset = [record for record in data if record[best_attr] == value]
            
            # If subset is empty, add a leaf node with the most common target value
            if not subset:
                most_common = target_counts.most_common(1)[0][0]
                tree.children[value] = Node(result=most_common)
            else:
                # Create a new list of attributes without the best one
                new_attributes = [attr for attr in attributes if attr != best_attr]
                
                # Recursively build the subtree
                subtree = self._id3(subset, new_attributes, target_attr)
                subtree.value = value
                tree.children[value] = subtree
                
        return tree
    
    def _entropy(self, data, target_attr):
        """
        Calculate the entropy of a dataset for the target attribute
        """
        value_counts = Counter([record[target_attr] for record in data])
        total = len(data)
        entropy = 0
        
        for count in value_counts.values():
            probability = count / total
            entropy -= probability * math.log2(probability)
            
        return entropy
    
    def _info_gain(self, data, attribute, target_attr):
        """
        Calculate the information gain of splitting on an attribute
        """
        # Calculate entropy before split
        entropy_before = self._entropy(data, target_attr)
        
        # Calculate weighted entropy after split
        values = set(record[attribute] for record in data)
        total = len(data)
        entropy_after = 0
        
        for value in values:
            subset = [record for record in data if record[attribute] == value]
            weight = len(subset) / total
            entropy_after += weight * self._entropy(subset, target_attr)
            
        # Return information gain
        return entropy_before - entropy_after
    
    def _choose_attribute(self, data, attributes, target_attr):
        """
        Choose the attribute with the highest information gain
        """
        gains = {attr: self._info_gain(data, attr, target_attr) for attr in attributes}
        return max(gains, key=gains.get)
    
    def _discretize_data(self, data, attributes, target_attr):
        """
        Discretize numeric attributes into bins
        """
        numeric_attrs = []
        
        # Identify numeric attributes
        for attr in attributes:
            if attr != target_attr and all(isinstance(record[attr], (int, float)) for record in data):
                numeric_attrs.append(attr)
        
        if not numeric_attrs:
            return data
            
        # Create a deep copy of the data
        discretized_data = [{k: v for k, v in record.items()} for record in data]
        
        # Discretize each numeric attribute
        for attr in numeric_attrs:
            # Extract values for this attribute
            values = [record[attr] for record in data]
            
            # Calculate bin thresholds (using numpy's percentile function)
            thresholds = []
            for i in range(1, self.bins):
                threshold = np.percentile(values, i * 100 / self.bins)
                thresholds.append(threshold)
            
            # Store thresholds for later use in prediction
            self.bin_thresholds[attr] = thresholds
            
            # Discretize values in the data
            for record in discretized_data:
                binned_value = self._get_bin(record[attr], thresholds)
                record[attr] = f"{attr}_{binned_value}"
                
        return discretized_data
    
    def _get_bin(self, value, thresholds):
        """
        Determine which bin a value belongs to based on thresholds
        """
        for i, threshold in enumerate(thresholds):
            if value <= threshold:
                return i
        return len(thresholds)
    
    def predict(self, record):
        """
        Predict the class of a single record
        """
        # Discretize numeric values if necessary
        if self.discretize_numeric:
            record = self._discretize_record(record)
            
        # Traverse the tree to make a prediction
        return self._predict_record(record, self.root)
    
    def _discretize_record(self, record):
        """
        Discretize numeric values in a single record
        """
        discretized = {k: v for k, v in record.items()}
        
        for attr, thresholds in self.bin_thresholds.items():
            if attr in record:
                binned_value = self._get_bin(record[attr], thresholds)
                discretized[attr] = f"{attr}_{binned_value}"
                
        return discretized
    
    def _predict_record(self, record, node):
        """
        Recursively traverse the tree to predict a record's class
        """
        # If we've reached a leaf node, return the result
        if node.result is not None:
            return node.result
        
        # Get the value of the attribute tested at this node
        attr_value = record.get(node.attribute)
        
        # If the value isn't in our training data, use the most common result from children
        if attr_value not in node.children:
            # Count the results in child nodes
            results = Counter()
            for child in node.children.values():
                if child.result is not None:
                    results[child.result] += 1
                    
            # Return the most common result
            return results.most_common(1)[0][0] if results else None
        
        # Otherwise, continue down the appropriate branch
        return self._predict_record(record, node.children[attr_value])
    
    def print_tree(self, node=None, indent=""):
        """
        Print the decision tree structure
        """
        if node is None:
            node = self.root
            
        # Print current node
        if node.result is not None:
            print(f"{indent}Result: {node.result}")
        else:
            print(f"{indent}Attribute: {node.attribute}")
            
            # Print child nodes
            for value, child in node.children.items():
                print(f"{indent}  Value: {value}")
                self.print_tree(child, indent + "    ")


class ConnectFourDataGenerator:
    """
    Generate training data for Connect Four by running MCTS simulations
    """
    def __init__(self, mcts_function):
        self.mcts_function = mcts_function
        
    def generate_dataset(self, num_samples=1000):
        """
        Generate a dataset of (board_state, best_move) pairs using MCTS
        
        Parameters:
        - num_samples: Number of samples to generate
        
        Returns:
        - List of dictionaries with board state features and best move
        """
        from game_structure import style as s
        from game_structure import game_engine as game
        import random
        
        dataset = []
        
        for _ in range(num_samples):
            # Create an empty board
            board = np.zeros((s.ROWS, s.COLUMNS), dtype=int)
            
            # Play a random number of moves to create diverse positions
            num_moves = random.randint(0, 20)
            player = s.FIRST_PLAYER_PIECE
            
            for _ in range(num_moves):
                # Get available moves
                available = game.available_moves(board)
                if available == -1 or game.winning_move(board, s.FIRST_PLAYER_PIECE) or game.winning_move(board, s.SECOND_PLAYER_PIECE):
                    break
                    
                # Make a random move
                col = random.choice(available)
                row = game.get_next_open_row(board, col)
                game.drop_piece(board, row, col, player)
                
                # Switch player
                player = s.SECOND_PLAYER_PIECE if player == s.FIRST_PLAYER_PIECE else s.FIRST_PLAYER_PIECE
            
            # If the game isn't over, get the best move using MCTS
            if not (game.winning_move(board, s.FIRST_PLAYER_PIECE) or 
                   game.winning_move(board, s.SECOND_PLAYER_PIECE) or
                   game.is_game_tied(board)):
                try:
                    best_move = self.mcts_function(board)
                    
                    # Convert board state to features
                    features = self._board_to_features(board)
                    features['best_move'] = best_move
                    
                    dataset.append(features)
                    
                    if len(dataset) % 10 == 0:
                        print(f"Generated {len(dataset)} samples")
                except Exception as e:
                    print(f"Error generating sample: {e}")
                    continue
        
        return dataset
    
    def _board_to_features(self, board):
        """
        Convert a board state to a dictionary of features
        """
        from game_structure import style as s
        
        features = {}
        
        # Count pieces in each column
        for col in range(s.COLUMNS):
            col_pieces = np.sum(board[:, col] != 0)
            features[f'col_{col}_count'] = int(col_pieces)
            
            # Record player pieces in each position of column
            for row in range(s.ROWS):
                if board[row, col] != 0:
                    features[f'pos_{row}_{col}'] = int(board[row, col])
                else:
                    features[f'pos_{row}_{col}'] = 0
        
        # Count pieces for each player
        features['player1_pieces'] = np.sum(board == s.FIRST_PLAYER_PIECE)
        features['player2_pieces'] = np.sum(board == s.SECOND_PLAYER_PIECE)
        
        # Difference in pieces (indicates whose turn it is)
        features['piece_diff'] = features['player1_pieces'] - features['player2_pieces']
        
        return features
    
    def save_dataset(self, dataset, filename):
        """
        Save dataset to a CSV file
        """
        if not dataset:
            print("No data to save!")
            return
            
        # Get all keys from the dataset
        fieldnames = list(dataset[0].keys())
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dataset)
            
        print(f"Dataset saved to {filename}")
    
    def load_dataset(self, filename):
        """
        Load dataset from a CSV file
        """
        dataset = []
        
        with open(filename, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Convert string values to appropriate types
                processed_row = {}
                for key, value in row.items():
                    try:
                        processed_row[key] = int(value)
                    except ValueError:
                        try:
                            processed_row[key] = float(value)
                        except ValueError:
                            processed_row[key] = value
                dataset.append(processed_row)
                
        print(f"Loaded {len(dataset)} samples from {filename}")
        return dataset


def process_iris_dataset(filename):
    """
    Process the Iris dataset for decision tree learning
    """
    import csv
    
    dataset = []
    attributes = []
    
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        
        # Get headers (attributes)
        attributes = next(reader)
        
        # Parse data
        for row in reader:
            record = {}
            for i, value in enumerate(row):
                try:
                    # Try to convert to numeric
                    record[attributes[i]] = float(value)
                except ValueError:
                    # Keep as string if not numeric
                    record[attributes[i]] = value
            dataset.append(record)
    
    print(f"Loaded {len(dataset)} records from {filename}")
    print(f"Attributes: {attributes}")
    
    return dataset, attributes


def evaluate_model(tree, test_data, target_attr):
    """
    Evaluate the decision tree on test data
    """
    correct = 0
    total = len(test_data)
    
    for record in test_data:
        prediction = tree.predict(record)
        if prediction == record[target_attr]:
            correct += 1
            
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total} correct)")
    return accuracy


def main():
    """
    Main function to demonstrate the decision tree implementation
    """
    # Part 1: Iris dataset
    print("Processing Iris dataset...")
    iris_data, iris_attrs = process_iris_dataset('Connect4-main/training/iris.csv')
    
    # Split into training and test sets (80% train, 20% test)
    import random
    random.shuffle(iris_data)
    split_idx = int(0.8 * len(iris_data))
    train_data = iris_data[:split_idx]
    test_data = iris_data[split_idx:]
    
    # Train decision tree
    tree = DecisionTree(discretize_numeric=True, bins=5)
    target_attr = iris_attrs[-1]  # Assuming last attribute is the target
    tree.fit(train_data, iris_attrs[:-1], target_attr)
    
    # Print tree structure
    print("\nIris Decision Tree:")
    tree.print_tree()
    
    # Evaluate model
    print("\nEvaluating on test data:")
    evaluate_model(tree, test_data, target_attr)
    
    # Part 2: Connect Four dataset
    print("\n\nGenerating Connect Four dataset...")
    from ai_alg import monte_carlo_ts as m
    
    # Generate dataset
    generator = ConnectFourDataGenerator(m.monte_carlo_ts)
    connect4_data = generator.generate_dataset(num_samples=100)  # Adjust number as needed
    
    # Save dataset
    generator.save_dataset(connect4_data, 'connect4_mcts.csv')
    
    # Split into training and test sets
    random.shuffle(connect4_data)
    split_idx = int(0.8 * len(connect4_data))
    train_data = connect4_data[:split_idx]
    test_data = connect4_data[split_idx:]
    
    # Train decision tree
    connect4_tree = DecisionTree(discretize_numeric=False)  # Already discretized by feature extraction
    connect4_attrs = [key for key in connect4_data[0].keys() if key != 'best_move']
    connect4_tree.fit(train_data, connect4_attrs, 'best_move')
    
    # Print tree structure (might be very large)
    print("\nConnect Four Decision Tree (partial view):")
    # connect4_tree.print_tree()  # Commented to avoid huge output
    
    # Evaluate model
    print("\nEvaluating on test data:")
    evaluate_model(connect4_tree, test_data, 'best_move')


if __name__ == "__main__":
    main()