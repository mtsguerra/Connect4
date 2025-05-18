import pickle
from graphviz import Digraph

# --- Load your model ---
with open("Connect4-main/training/data/connect4_tree_model_300games.pkl", "rb") as f:
    model = pickle.load(f)

root = getattr(model, "root", model)

# --- Create Graphviz graph ---
dot = Digraph(comment="Connect4 Decision Tree")
node_counter = 0

dot.attr(rankdir="TB")  # TB = top-to-bottom (instead of LR = left-to-right)

def add_nodes_edges(node, parent_id=None, edge_label=""):
    global node_counter
    node_id = f"n{node_counter}"
    node_counter += 1

    if node.is_leaf:
        label = f"Predict: {node.prediction}"
        dot.node(node_id, label, shape="box", style="filled", color="lightblue")
    else:
        label = f"{node.feature}"
        dot.node(node_id, label, shape="ellipse")

    if parent_id is not None:
        dot.edge(parent_id, node_id, label=str(edge_label))

    if not node.is_leaf:
        for val, child in node.children.items():
            add_nodes_edges(child, node_id, edge_label=val)

add_nodes_edges(root)

# --- Render the tree ---
dot.render("connect4_tree", format="png", cleanup=False)
print("✅ Graph rendered as 'connect4_tree.png'")
