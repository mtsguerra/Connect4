import numpy as np
import pandas as pd
import os
import sys
import time
import pickle
from collections import Counter
from math import log

# Adiciona o diretório raiz do projeto ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Node:
    
    def __init__(self):
        self.feature = None      
        self.value = None        
        self.children = {}       
        self.is_leaf = False     
        self.prediction = None   
        self.depth = 0     

class ID3Tree:
    """Implementação mais simples da árvore de decisão ID3"""
    
    def __init__(self, max_depth=10):
        """Inicializa a árvore com profundidade máxima flexível"""
        
        self.root = None
        self.max_depth = max_depth
    
    def fit(self, X, y):
        """Constrói a árvore de decisão a partir dos dados do treino"""
        
        print("Treinando árvore de decisão ID3...")
        start_time = time.time()
        
        X = np.array(X)
        y = np.array(y)
        
        features = list(range(X.shape[1]))
        
        # Constrói a árvore
        self.root = self._build_tree(X, y, features, depth=0)
        
        print(f"Treinamento concluído em {time.time() - start_time:.2f} segundos")
        return self
    
    def predict(self, X):
        """Prevê classes para amostras em X"""
        
        X = np.array(X)
        return np.array([self._predict_sample(x, self.root) for x in X])
    
    def _predict_sample(self, x, node):
        """Prevê a classe para uma única amostra"""
        
        if node.is_leaf:
            return node.prediction
        
        feature_value = x[node.feature]
        
        if feature_value not in node.children:
            predictions = [child.prediction for child in node.children.values() if child.is_leaf]
            if not predictions:
                return list(node.children.values())[0].prediction
            else:
                return Counter(predictions).most_common(1)[0][0]
        
        return self._predict_sample(x, node.children[feature_value])
    
    def _build_tree(self, X, y, features, depth):
        """Constrói a árvore de decisão com recursão"""
        
        node = Node()
        node.depth = depth
        
        # Se todas amostras têm a mesma classe, cria nó folha
        if len(np.unique(y)) == 1:
            node.is_leaf = True
            node.prediction = y[0]
            return node
        
        # Se acabar os atributos ou atingir profundidade máxima, cria uma folha
        if len(features) == 0 or depth >= self.max_depth:
            node.is_leaf = True
            # Usa classe mais comum
            node.prediction = Counter(y).most_common(1)[0][0]
            return node
        
        # Encontra o melhor atributo para dividir, com o maior ganho de informação
        best_feature = self._find_best_feature(X, y, features)
        node.feature = best_feature
        
        # Obtém valores únicos do melhor atributo
        unique_values = np.unique(X[:, best_feature])
        
        # Cria filhos para cada valor
        new_features = features.copy()
        new_features.remove(best_feature)
        
        # Checa se há amostras suficientes para dividir
        if len(unique_values) <= 1:
            node.is_leaf = True
            node.prediction = Counter(y).most_common(1)[0][0]
            return node
        
        # Cria nós filhos
        for value in unique_values:
            # Seleciona amostras com esse valor
            indices = np.where(X[:, best_feature] == value)[0]
            
            if len(indices) == 0:
                continue
            
            # Cria nó filho
            child = self._build_tree(X[indices], y[indices], new_features, depth + 1)
            node.children[value] = child
            
        # Se não criou filhos, cria folha em último caso
        if not node.children:
            node.is_leaf = True
            node.prediction = Counter(y).most_common(1)[0][0]
            
        return node
    
    def _find_best_feature(self, X, y, features):
        """Encontra o atributo com o maior ganho de informação"""
        
        best_gain = -1
        best_feature = None
        base_entropy = self._entropy(y)
        
        for feature in features:
            if len(np.unique(X[:, feature])) <= 1:
                continue
            gain = self._information_gain(X, y, feature, base_entropy)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        if best_feature is None and features:
            best_feature = features[0]
            
        return best_feature
    
    def _entropy(self, y):
        """Calcula a entropia da distribuição de classes"""
        
        counts = Counter(y)
        total = len(y)
        entropy = 0
        
        for count in counts.values():
            p = count / total
            entropy -= p * log(p, 2)
            
        return entropy
    
    def _information_gain(self, X, y, feature, base_entropy):
        """Calcula o ganho de informação de um atributo"""
        
        values = X[:, feature]
        unique_values = np.unique(values)
        
        # Calcula entropia ponderada
        weighted_entropy = 0
        total_samples = len(y)
        
        for value in unique_values:
            # Seleciona amostras com esse valor
            indices = np.where(values == value)[0]
            subset_y = y[indices]
            
            # Peso e entropia
            weight = len(subset_y) / total_samples
            weighted_entropy += weight * self._entropy(subset_y)
        
        # Ganho de informação
        return base_entropy - weighted_entropy
    
    def save(self, filename):
        """Salva o modelo em arquivo"""
        
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
            
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Modelo salvo em {filename}")
    
    @staticmethod
    def load(filename):
        """Carrega modelo de arquivo"""
        
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Modelo carregado de {filename}")
        return model
    
    def print_tree(self, max_depth=3):
        """Imprime uma representação textual da árvore"""
        
        if self.root is None:
            print("Árvore ainda não construída")
            return
        
        self._print_node(self.root, "", True, max_depth)
    
    def _print_node(self, node, indent, is_last, max_depth=None):
        """Imprime um nó e seus filhos"""
        
        if max_depth is not None and node.depth > max_depth:
            return
            
        if node.is_leaf:
            print(f"{indent}{'└── ' if is_last else '├── '}Predição: {node.prediction}")
        else:
            print(f"{indent}{'└── ' if is_last else '├── '}Atributo: {node.feature}")
            
            child_keys = list(node.children.keys())
            for i, value in enumerate(child_keys):
                child = node.children[value]
                is_last_child = (i == len(child_keys) - 1)
                new_indent = indent + ("    " if is_last else "│   ")
                print(f"{new_indent}{'└── ' if is_last_child else '├── '}Valor: {value}")
                
                self._print_node(child, new_indent + "    ", is_last_child, max_depth)

def load_and_split_data(file_path, test_size=0.2, random_seed=42):
    """Carrega os dados do CSV e divide em treino/teste"""
    
    print(f"Carregando dataset de {file_path}...")
    
    # Checa se o arquivo existe
    if not os.path.exists(file_path):
        print(f"Erro: Arquivo {file_path} não encontrado")
        return None, None, None, None
    
    # Carrega os dados
    df = pd.read_csv(file_path)
    print(f"Shape do dataset: {df.shape}")
    
    # Separa atributos e alvo
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Semente para reprodutibilidade
    np.random.seed(random_seed)
    
    # Gera índices para split treino/teste
    indices = np.random.permutation(len(X))
    test_size_int = int(test_size * len(X))
    test_indices = indices[:test_size_int]
    train_indices = indices[test_size_int:]
    
    # Split
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    print(f"Treino: {len(X_train)} amostras")
    print(f"Teste: {len(X_test)} amostras")
    
    return X_train, y_train, X_test, y_test


def evaluate_model(y_true, y_pred):
    """Calcula e imprime métricas de desempenho do modelo"""
    
    # Faz cópia dos arrays para manter os originais
    y_true_clean = np.array(y_true).copy()
    y_pred_clean = np.array(y_pred).copy()
    
    # Substitui valores None nas predições por um valor especial (-1)
    none_indices = [i for i, val in enumerate(y_pred_clean) if val is None]
    if none_indices:
        print(f"Aviso: Encontrado(s) {len(none_indices)} predições None. Substituindo por valor especial para avaliação.")
        for i in none_indices:
            y_pred_clean[i] = -1
    
    # Calcula a acurácia apenas para predições válidas
    valid_indices = [i for i, val in enumerate(y_pred) if val is not None]
    
    if len(valid_indices) > 0:
        accuracy = np.mean(y_true_clean[valid_indices] == y_pred_clean[valid_indices]) * 100
        print(f"Acurácia: {accuracy:.2f}% (excluindo predições None)")
        
        # Calcula a acurácia geral contando None como erro
        overall_accuracy = np.mean(y_true == y_pred) * 100
        print(f"Acurácia geral: {overall_accuracy:.2f}% (contando None como erro)")
    else:
        print("Aviso: Todas as predições são None!")
        accuracy = 0
    
    # Distribuição das classes previstas
    class_dist = {}
    for pred in y_pred:
        pred_key = str(pred)  # Converte para string para lidar com None
        if pred_key not in class_dist:
            class_dist[pred_key] = 0
        class_dist[pred_key] += 1
    
    print("\nDistribuição das classes previstas:")
    for col, count in sorted(class_dist.items()):
        print(f"Coluna {col}: {count} ({count/len(y_pred)*100:.2f}%)")
    
    # Todas classes únicas (exceto None)
    true_classes = set([y for y in y_true if y is not None])
    pred_classes = set([y for y in y_pred if y is not None])
    classes = sorted(true_classes.union(pred_classes))
    
    # Matriz de confusão (excluindo predições None)
    conf_matrix = np.zeros((len(classes), len(classes)), dtype=int)
    
    for i in range(len(y_true_clean)):
        if y_pred_clean[i] == -1:  # Pula predições None
            continue
            
        # Índices nas nossas classes ordenadas
        true_idx = classes.index(y_true_clean[i])
        pred_idx = classes.index(y_pred_clean[i])
        conf_matrix[true_idx, pred_idx] += 1
    
    print("\nMatriz de Confusão (excluindo predições None):")
    header = " " * 10
    for c in classes:
        header += f"Pred {c:<5} "
    print(header)
    
    for i, c in enumerate(classes):
        row = f"True {c:<5} "
        for j in range(len(classes)):
            row += f"{conf_matrix[i, j]:<10} "
        print(row)
    
    # Métricas por classe
    for c in classes:
        c_idx = classes.index(c)
        true_positives = conf_matrix[c_idx, c_idx]
        all_predicted = np.sum(conf_matrix[:, c_idx])
        all_actual = np.sum(conf_matrix[c_idx, :])
        
        precision = true_positives / all_predicted if all_predicted > 0 else 0
        recall = true_positives / all_actual if all_actual > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nClasse {c}:")
        print(f"  Precisão: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")
    
    return accuracy, conf_matrix


def train_connect4_model(data_path=None, max_depth=10, save_path=None):
    """Treina a decision_tree"""
    
    # Define caminhos padrão se não fornecido
    if data_path is None:
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "connect4_dataset_mixed.csv")
    
    if save_path is None:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "connect4_tree_model_mixed.pkl")
    
    # Carrega e divide o dataset
    X_train, y_train, X_test, y_test = load_and_split_data(data_path, test_size=0.2)
    
    if X_train is None:
        print("Erro ao carregar dados. Encerrando.")
        return None
    
    # Treina o modelo
    model = ID3Tree(max_depth=max_depth)
    model.fit(X_train, y_train)
    
    # Avalia no treino
    print("\nAvaliando no treino:")
    y_train_pred = model.predict(X_train)
    train_acc, _ = evaluate_model(y_train, y_train_pred)
    
    # Avalia no teste
    print("\nAvaliando no teste:")
    y_test_pred = model.predict(X_test)
    test_acc, _ = evaluate_model(y_test, y_test_pred)
    
    # Salva modelo
    if save_path:
        model.save(save_path)
    
    # Imprime estrutura da árvore (até 3 níveis)
    print("\nEstrutura da Árvore (até 3 níveis):")
    model.print_tree(max_depth=3)
    
    return model

def train_iris_model(iris_path=None):
    """Treina a decision_tree utilizando a Iris.csv"""
    
    # Caminho padrão, se não fornecido
    if iris_path is None:
        iris_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "iris.csv"
        )
        
    print(f"Carregando dataset iris de {iris_path}...")
    
    # Checa se o arquivo existe
    if not os.path.exists(iris_path):
        print(f"Erro: Dataset iris não encontrado em {iris_path}")
        return None
        
    # Carrega dados
    try:
        df = pd.read_csv(iris_path)
        print(f"Dataset iris carregado com shape: {df.shape}")
        
        # Extrai atributos
        feature_cols = [col for col in df.columns if col.lower() not in ['id', 'class']]
        X = df[feature_cols].values
        
        class_mapping = {
            'Iris-setosa': 0,
            'Iris-versicolor': 1,
            'Iris-virginica': 2
        }
        
        # Coluna de classe
        class_col = [col for col in df.columns if 'class' in col.lower()][0]
        y = df[class_col].map(class_mapping).values
        
        # Checa se o mapeamento funcionou
        if np.any(pd.isna(y)):
            print("Aviso: Alguns rótulos de classe não foram mapeados.")
            print("Classes únicas no dataset:", df[class_col].unique())
            # Tenta corrigir valores não mapeados
            unmapped = df.loc[pd.isna(df[class_col].map(class_mapping)), class_col].unique()
            for i, cls in enumerate(unmapped):
                print(f"Mapeando '{cls}' para {i}")
                y[df[class_col] == cls] = i
        
        # Nomes dos atributos
        feature_names = feature_cols
        print(f"Atributos: {feature_names}")
        print(f"Classes: {list(class_mapping.keys())}")
        
        # Split - sem scikit-learn
        np.random.seed(42)
        indices = np.random.permutation(len(X))
        train_indices = indices[:int(0.8 * len(X))]
        test_indices = indices[int(0.8 * len(X)):]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        
        print(f"Amostras de treino: {len(X_train)}")
        print(f"Amostras de teste: {len(X_test)}")
        
        for i in range(X_train.shape[1]):
            values = X_train[:, i]
            thresholds = [
                np.percentile(values, 33),
                np.percentile(values, 66)
            ]
            
            X_train[:, i] = np.digitize(X_train[:, i], thresholds)
            X_test[:, i] = np.digitize(X_test[:, i], thresholds)
        
        # Treina modelo
        model = ID3Tree(max_depth=4)
        model.fit(X_train, y_train)
        
        # Avalia
        print("\nResultados no Dataset Iris:")
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_acc = np.mean(train_pred == y_train) * 100
        test_acc = np.mean(test_pred == y_test) * 100
        
        print(f"Acurácia treino: {train_acc:.2f}%")
        print(f"Acurácia teste: {test_acc:.2f}%")
        
        # Classes
        print("\nMapeamento de classes:")
        for cls, idx in class_mapping.items():
            print(f"{idx}: {cls}")
        
        # Imprime árvore
        print("\nÁrvore de Decisão Iris:")
        model.print_tree()
        
        # Salva modelo se diretório existir
        try:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
            os.makedirs(output_dir, exist_ok=True)
            model.save(os.path.join(output_dir, "iris_tree.pkl"))
        except:
            print("Não foi possível salvar o modelo")
        
        return model
        
    except Exception as e:
        print(f"Erro ao carregar dataset iris: {e}")
        import traceback
        traceback.print_exc()
        return None
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Treina árvore de decisão ID3 para Connect4")
    parser.add_argument("--data", type=str, help="Caminho para o CSV do dataset Connect4")
    parser.add_argument("--iris-data", type=str, default="training/iris.csv", help="Caminho para o CSV do dataset iris")
    parser.add_argument("--depth", type=int, default=10, help="Profundidade máxima da árvore")
    parser.add_argument("--iris", action="store_true", help="Treina no dataset iris primeiro")
    parser.add_argument("--output", type=str, help="Caminho para salvar o modelo")
    
    args = parser.parse_args()
    
    # Treina no iris se solicitado
    if args.iris:
        print("=== Treinando no Dataset Iris ===")
        train_iris_model(iris_path=args.iris_data)
        print("\n")
    
    # Treina no Connect4
    print("=== Treinando no Dataset Connect4 ===")
    train_connect4_model(
        data_path=args.data,
        max_depth=args.depth,
        save_path=args.output
    )
