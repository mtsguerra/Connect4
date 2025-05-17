import numpy as np
from collections import Counter

class ID3Tree:
    def __init__(self, max_depth=None):
        """
        Inicializa a árvore de decisão ID3.
        :param max_depth: Profundidade máxima da árvore.
        """
        self.tree = None
        self.max_depth = max_depth

    def entropy(self, y):
        counts = Counter(y)
        total = len(y)
        entropy_value = 0
        
        for count in counts.values():
            p = count / total
            entropy_value += -p * np.log2(p) if p > 0 else 0
        
        return entropy_value
    
    def info_gain(self, y, y_left, y_right):
        p = len(y_left) / len(y)
        gain = self.entropy(y) - (p * self.entropy(y_left) + (1 - p) * self.entropy(y_right))
        return gain
    
    def best_split(self, X, y):
        best_gain = 0
        best_attr = None
        best_splits = None 
        
        for col in X.columns:
            values = X[col].unique()
            for val in values:
                left_mask = X[col] == val
                right_mask = ~left_mask
                
                y_left, y_right = y[left_mask], y[right_mask]
                
                gain = self.info_gain(y, y_left, y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_attr = (col, val)
                    best_splits = (X[left_mask], y[left_mask], X[right_mask], y[right_mask])
                    
        return best_attr, best_splits
    
    def build_tree(self, X, y, depth=0):
        """
        Constrói a árvore de decisão recursivamente.
        Args:
            X (DataFrame): Atributos das amostras.
            y (Series): Rótulos das amostras.
            depth (int): Profundidade atual.
        """
        # Caso 1: Todos os rótulos são iguais
        if len(set(y)) == 1:
            return y.iloc[0]

        # Caso 2: Sem mais atributos ou profundidade máxima atingida
        if X.empty or (self.max_depth is not None and depth >= self.max_depth):
            return y.mode()[0]

        # Escolhe o melhor atributo para dividir
        attr, splits = self.best_split(X, y)
        if attr is None:
            return y.mode()[0]

        # Caso 3: Divisão recursiva com controle de profundidade
        left_branch = self.build_tree(splits[0], splits[1], depth + 1)
        right_branch = self.build_tree(splits[2], splits[3], depth + 1)

        return {attr: {'left': left_branch, 'right': right_branch}}
        
    def fit(self, X, y):
        """
        Treina a árvore de decisão com os dados fornecidos.
        Args:
            X (DataFrame): Atributos das amostras.
            y (Series): Rótulos das amostras.
        """
        self.tree = self.build_tree(X, y)

    def predict_one(self, x, tree):
        """
        Prediz a classe para uma única amostra.
        Args:
            x (Series): Atributos da amostra.
            tree (dict): Árvore de decisão treinada.
        Returns:
            int: Classe prevista.
        """
        if not isinstance(tree, dict):
            return tree

        attr, branches = list(tree.items())[0]
        value = x[attr[0]]  

        # Vai para o ramo esquerdo ou direito com base no valor
        if value == attr[1]:
            return self.predict_one(x, branches['left'])
        else:
            return self.predict_one(x, branches['right'])
        
    def predict(self, X):
        """
        Prediz as classes para múltiplas amostras.
        Args:
            X (DataFrame): Atributos das amostras.
        Returns:
            Series: Classes previstas.
        """
        return X.apply(lambda x: self.predict_one(x, self.tree), axis=1)