import pickle
from graphviz import Digraph

# Carrega o modelo de árvore de decisão treinado
with open("Connect4-main/training/data/connect4_tree_model_300games.pkl", "rb") as f:
    model = pickle.load(f)

# Normaliza acesso à raiz da árvore
root = getattr(model, "root", model)

# Cria o grafo Graphviz para visualizar a árvore
dot = Digraph(comment="Árvore de Decisão Connect4")
contador_nos = 0

# Configuração para o grafo crescer de cima para baixo
dot.attr(rankdir="TB")

def adicionar_nos_e_arestas(no, id_pai=None, rotulo_aresta=""):
    """Adiciona recursivamente nós e arestas no grafo para cada nó da árvore"""
    
    global contador_nos
    id_no = f"n{contador_nos}"
    contador_nos += 1

    # Se for folha, destaque e mostre previsão
    if no.is_leaf:
        label = f"Previsão: {no.prediction}"
        dot.node(id_no, label, shape="box", style="filled", color="lightblue")
    else:
        label = f"Pergunta: {no.feature}"
        dot.node(id_no, label, shape="ellipse")

    # Conecta ao nó pai, se existir
    if id_pai is not None:
        dot.edge(id_pai, id_no, label=str(rotulo_aresta))

    # Se não for folha, processa filhos
    if not no.is_leaf:
        for valor, filho in no.children.items():
            adicionar_nos_e_arestas(filho, id_no, rotulo_aresta=valor)

# Começa a recursão pela raiz
adicionar_nos_e_arestas(root)

# Renderiza a árvore para arquivo PNG
saida = dot.render("connect4_tree", format="png", cleanup=False)
print("Árvore de decisão renderizada como 'connect4_tree.png'")
