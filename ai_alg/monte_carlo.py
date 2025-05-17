import time, math, numpy as np, random, itertools
from game_structure import style as s
from game_structure import game_engine as game
from math import sqrt, log
from ai_alg import heuristic as h

class Node:
    """
    Represents a node in the Monte Carlo search tree.
    Each node corresponds to a game state (board configuration).
    """
    def __init__(self, board, last_player, parent=None) -> None:
        self.board = board  # Current board state
        self.parent = parent  # Parent node in the tree
        self.children = []  # List of child nodes
        self.visits = 0  # Number of times this node has been visited
        self.wins = 0  # Number of wins achieved from this node
        # Determine current player based on who played last
        self.current_player = 1 if last_player == 2 else 2

    def __str__(self) -> str:
        """String representation of node for debugging"""
        string = "Wins: " + str(self.wins) + '\n'
        string += "Total visits: " + str(self.visits) + '\n'
        string += "UCB score: " + str(self.ucb()) + '\n'
        string += "Win probability: " + str(self.score()) + '\n'
        return string

    def add_children(self) -> None:
        """
        Generate all possible moves from current board state and 
        add them as children to this node
        """
        # If children already exist or no moves are available, do nothing
        if (len(self.children) != 0) or (game.available_moves(self.board) == -1):
            return
            
        # Generate a child node for each possible move
        for col in game.available_moves(self.board):
            # Create a copy of the board with the new move applied
            if self.current_player == s.FIRST_PLAYER_PIECE:
                copy_board = game.simulate_move(self.board, s.SECOND_PLAYER_PIECE, col)
            else:
                copy_board = game.simulate_move(self.board, s.FIRST_PLAYER_PIECE, col)
                
            # Add the new node and the column that generated it to the children list
            self.children.append((Node(board=copy_board, 
                                       last_player=self.current_player, 
                                       parent=self), col))

    def select_children(self):
        """
        Randomly select up to 4 children to explore
        This limits branching factor to improve efficiency
        """
        if (len(self.children) > 4):
            return random.sample(self.children, 4)
        return self.children

    def ucb(self) -> float:
        """
        Calculate Upper Confidence Bound for node selection
        Balances exploitation (known good moves) with exploration (unexplored moves)
        """
        if self.visits == 0:
            return float('inf')  # Unvisited nodes have highest priority
            
        # UCB formula: wins/visits + C * sqrt(ln(parent_visits)/visits)
        exploitation = self.wins / self.visits
        exploration = sqrt(2) * sqrt(log(self.parent.visits) / self.visits)
        return exploitation + exploration
        
    def score(self) -> float:
        """Calculate win rate (wins/visits) for this node"""
        if self.visits == 0:
            return 0
        return self.wins / self.visits


class monte_carlo:
    """
    Monte Carlo Tree Search implementation for Connect Four
    Uses UCB1 for node selection and heuristic-guided simulation
    """
    def __init__(self, root: Node) -> None:
        self.root = root

    def start(self, max_time: int):
        """
        Initialize search by performing initial simulations on root's children
        Then run the main MCTS algorithm
        """
        # Generate children for the root node
        self.root.add_children()
        
        # Perform initial simulations on each child
        for child in self.root.children:
            # If we find an immediate winning move, return it
            if game.winning_move(child[0].board, s.SECOND_PLAYER_PIECE):
                return child[1]
                
            # Run 6 simulations for each child to get initial statistics
            for _ in range(6):
                result = self.rollout(child[0])
                self.back_propagation(child[0], result)
                
        # Start the main MCTS algorithm
        return self.search(max_time)

    def search(self, max_time: int) -> int:
        """
        Main MCTS algorithm: select, expand, simulate, backpropagate
        Runs until time limit is reached
        """
        start_time = time.time()
        
        # Continue until time limit is reached
        while time.time() - start_time < max_time:
            # Select a leaf node to explore
            selected_node = self.select(self.root)
            
            if selected_node.visits == 0:
                # If node hasn't been visited, simulate from it directly
                result = self.rollout(selected_node)
                self.back_propagation(selected_node, result)
            else:
                # If node has been visited, expand it and simulate its children
                selected_children = self.expand(selected_node)
                for child in selected_children:
                    result = self.rollout(child[0])
                    self.back_propagation(child[0], result)
                    
        # Return the best move based on statistics gathered
        return self.best_move()

    def select(self, node: Node) -> Node:
        """
        Select the most promising leaf node to explore
        Uses UCB to balance exploration and exploitation
        """
        if node.children == []:
            # Base case: node is a leaf
            return node
        else:
            # Select best child based on UCB and continue recursively
            node = self.best_child(node)
            return self.select(node)

    def best_child(self, node: Node) -> Node:
        """Select child with highest UCB score"""
        best_child = None
        best_score = float('-inf')
        
        for (child, _) in node.children:
            ucb = child.ucb()
            if ucb > best_score:
                best_child = child
                best_score = ucb
                
        return best_child

    def back_propagation(self, node: Node, result: int) -> None:
        """
        Update statistics for all nodes from leaf to root
        Increment visit count and win count if applicable
        """
        while node:
            node.visits += 1  # Increment visit count
            
            # Increment win count if this node's player won the simulation
            if node.current_player == result:
                node.wins += 1
                
            # Move up to parent node
            node = node.parent

    def expand(self, node: Node) -> Node:
        """
        Add all possible child nodes and select some for simulation
        """
        node.add_children()
        return node.select_children()

    def rollout(self, node: Node) -> int:
        """
        Simulate a game from the current node using heuristic-guided play
        Returns the winner of the simulation
        """
        board = node.board.copy()
        max_depth = 6  # Limit simulation depth for efficiency
        players = itertools.cycle([s.SECOND_PLAYER_PIECE, s.FIRST_PLAYER_PIECE])
        current_player = next(players)

        for _ in range(max_depth):
            # Check if game is already decided
            if game.winning_move(board, s.SECOND_PLAYER_PIECE) or game.winning_move(board, s.FIRST_PLAYER_PIECE):
                break
            if game.is_game_tied(board):
                return 0  # Tie game
                
            current_player = next(players)
            moves = game.available_moves(board)
            
            if moves == -1:
                return 0  # No moves available
                
            # Use heuristic to choose best move (not random)
            best_move = max(
                moves,
                key=lambda col: h.calculate_board_score(
                    game.simulate_move(board, current_player, col), 
                    s.SECOND_PLAYER_PIECE, 
                    s.FIRST_PLAYER_PIECE
                )
            )
            
            # Apply the selected move
            board = game.simulate_move(board, current_player, best_move)

        return current_player

    def best_move(self) -> int:
        """
        Select the best move based on win rates of root's children
        If multiple moves have the same score, choose randomly
        """
        max_score = float('-inf')
        scores = {}  # Store (column, score) pairs
        columns = []  # Store columns with the best score
        
        # Calculate scores for all possible moves
        for (child, col) in self.root.children:
            score = child.score()
            print(f"Column: {col}")
            print(child)
            
            if score > max_score:
                max_score = score
                
            scores[col] = score
            
        # Collect all moves with the best score
        for col, score in scores.items():
            if score == max_score:
                columns.append(col)
                
        # Choose randomly among best moves
        return random.choice(columns)


def mcts(board: np.ndarray) -> int:
    """
    Entry point function that runs Monte Carlo Tree Search
    Returns the best column (0-indexed) to play
    """
    root = Node(board=board, last_player=s.SECOND_PLAYER_PIECE)
    mc = monte_carlo(root)
    column = mc.start(3)  # Run MCTS for 3 seconds
    print(column + 1)  # Print 1-indexed column for human readability
    return column
