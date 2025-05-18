import time, math, numpy as np, random, itertools
from game_structure import style as s
from game_structure import game_engine as game
from math import sqrt, log
from ai_alg import heuristic as h

class Node:
    """Representa um nó na árvore de busca Monte Carlo, onde cada nó representa uma configuração diferente do tabuleiro"""
    
    def __init__(self, board, last_player, parent=None) -> None:
        self.board = board  # Estado atual do tabuleiro
        self.parent = parent  # Nó pai na árvore
        self.children = []  # Lista de nós filhos
        self.visits = 0  # Número de vezes que este nó foi visitado
        self.wins = 0  # Número de vitórias alcançadas a partir deste nó
        # Determina o jogador atual com base em que jogou por ultimo
        self.current_player = 1 if last_player == 2 else 2

    def __str__(self) -> str:
        """Representação em string do nó para depuração"""
        
        string = "Vitórias: " + str(self.wins) + '\n'
        string += "Total de visitas: " + str(self.visits) + '\n'
        string += "Pontuação UCB: " + str(self.ucb()) + '\n'
        string += "Probabilidade de vitória: " + str(self.score()) + '\n'
        return string

    def add_children(self) -> None:
        """Gera todos possíveis movimentos no tabuleiro atual, adicionando-os como filhos deste nó"""
        
        if (len(self.children) != 0) or (game.available_moves(self.board) == -1):
            return
            
        for col in game.available_moves(self.board):
            # Cria uma cópia do tabuleiro com a nova jogada aplicada
            if self.current_player == s.FIRST_PLAYER_PIECE:
                copy_board = game.simulate_move(self.board, s.SECOND_PLAYER_PIECE, col)
            else:
                copy_board = game.simulate_move(self.board, s.FIRST_PLAYER_PIECE, col)
                
            # Adiciona o novo nó e a coluna que o gerou à lista de filhos
            self.children.append((Node(board=copy_board, last_player=self.current_player, parent=self), col))

    def select_children(self):
        """Seleciona aleatoriamente no máximo 4 filhos para apurar a busca, melhorando assim a sua eficiência"""
        
        if (len(self.children) > 4):
            return random.sample(self.children, 4)
        return self.children

    def ucb(self) -> float:
        """Cacula o UCB (Upper Confidence Bound) para escolher os nós, equilibrando assim as escolhas aprofundadas e as não"""
        
        if self.visits == 0:
            return float('inf')  # Nós não visitados têm prioridade máxima
            
        # Fórmula UCB: wins/visits + C * sqrt(ln(parent_visits)/visits)
        exploitation = self.wins / self.visits
        exploration = sqrt(2) * sqrt(log(self.parent.visits) / self.visits)
        return exploitation + exploration
        
    def score(self) -> float:
        """Calcula a taxa de vitórias para este nó"""
        if self.visits == 0:
            return 0
        return self.wins / self.visits


class monte_carlo:
    """Implementação do algoritmo de Monte Carlo TS, usando UCB1 para escolher os nós e simulando usando heurística"""
    
    def __init__(self, root: Node) -> None:
        self.root = root

    def start(self, max_time: int):
        """Inicia a busca realizando simulações iniciais nos filhos da raiz, executando em seguida o algoritmo mcts"""
        
        self.root.add_children()   
        for child in self.root.children:
            if game.winning_move(child[0].board, s.SECOND_PLAYER_PIECE):
                return child[1]  
            for _ in range(6):
                result = self.rollout(child[0])
                self.back_propagation(child[0], result)
                
        return self.search(max_time)

    def search(self, max_time: int) -> int:
        """Executa o algoritmo princiapl do mcts até o limite de tempo"""
        
        start_time = time.time()
        while time.time() - start_time < max_time:
            selected_node = self.select(self.root)
            
            # Analisa se o nó foi ou não visitado, mudando ou não de onde partir
            if selected_node.visits == 0:
                result = self.rollout(selected_node)
                self.back_propagation(selected_node, result)
            else:
                selected_children = self.expand(selected_node)
                for child in selected_children:
                    result = self.rollout(child[0])
                    self.back_propagation(child[0], result)
                                   
        return self.best_move()

    def select(self, node: Node) -> Node:
        """Seleciona o nó com maiores chances, utilizando UCB para equilibrar"""
        
        if node.children == []:
            return node
        else:
            node = self.best_child(node)
            return self.select(node)

    def best_child(self, node: Node) -> Node:
        best_child = None
        best_score = float('-inf')
        
        for (child, _) in node.children:
            ucb = child.ucb()
            if ucb > best_score:
                best_child = child
                best_score = ucb
                
        return best_child

    def back_propagation(self, node: Node, result: int) -> None:
        """Atualiza os dados para todos os nós da árvore, incrementando visit count e possívelmente win count"""
        
        while node:
            node.visits += 1
            if node.current_player == result:
                node.wins += 1
            node = node.parent

    def expand(self, node: Node) -> Node:
        """Adiciona todos os possíveis nós filhos e seleciona alguns"""
        
        node.add_children()
        return node.select_children()

    def rollout(self, node: Node) -> int:
        """Simula uma possível partida a partir do nó atual, utilizando de jogadas guiadas por heurística, retornando assim o vencedor da simulação"""
        
        board = node.board.copy()
        max_depth = 6
        players = itertools.cycle([s.SECOND_PLAYER_PIECE, s.FIRST_PLAYER_PIECE])
        current_player = next(players)

        for _ in range(max_depth):
            if game.winning_move(board, s.SECOND_PLAYER_PIECE) or game.winning_move(board, s.FIRST_PLAYER_PIECE):
                break
            if game.is_game_tied(board):
                return 0   
            current_player = next(players)
            moves = game.available_moves(board) 
            if moves == -1:
                return 0 
            # Usa heurística para escolher a melhor jogada (não aleatória)
            best_move = max(
                moves,
                key=lambda col: h.calculate_board_score(
                    game.simulate_move(board, current_player, col), 
                    s.SECOND_PLAYER_PIECE, 
                    s.FIRST_PLAYER_PIECE
                )
            )
            
            board = game.simulate_move(board, current_player, best_move)

        return current_player

    def best_move(self) -> int:
        """Seleciona as melhores opções, baseado na taxa de vitória dos filhos da raíz, possívelmente com a mesma pontuação, "desempatando" aleatoriamente"""
        
        max_score = float('-inf')
        scores = {}  # Armazena os pares de colunas e suas respectivas pontuações
        columns = []  # Armazena as colunas com a melhor pontuação
        
        # Calcula as pontuações possíveis
        for (child, col) in self.root.children:
            score = child.score()
            print(f"Coluna: {col}")
            print(child)
            
            if score > max_score:
                max_score = score
                
            scores[col] = score
            
        # Armazena todos os movimentos que resultam com a melhor pontuação
        for col, score in scores.items():
            if score == max_score:
                columns.append(col)
                
        # Escolhe aleatoriamente entre os melhores movimentos
        return random.choice(columns)


def mcts(board: np.ndarray) -> int:
    """Função que executa a busca mcts e retorna a melhor coluna para se jogar"""
    
    root = Node(board=board, last_player=s.SECOND_PLAYER_PIECE)
    mc = monte_carlo(root)
    column = mc.start(3)  # Executa o MCTS por 3 segundos
    print(column + 1)
    return column
