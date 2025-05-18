import pygame
import itertools
import sys
from game_structure import style as s
from game_structure import Board
from game_structure import game_engine as game
from dataclasses import dataclass
from game_structure.ai_game import run_ai_vs_ai_game

@dataclass
class Interface:
    rows: int = s.ROWS
    columns: int = s.COLUMNS
    pixels: int = s.SQUARE_SIZE
    width: int = s.WIDTH
    height: int = s.HEIGHT
    rad: float = s.RADIUS_PIECE
    size: tuple = (width, height)
    screen: any = pygame.display.set_mode(size)
    pygame.display.set_caption("Connect4")

    def starting_game(self, brd: Board):
        """Inicia o programa para rodar o jogo"""
        
        pygame.init()
        self.draw_menu()
        game_mode = self.choose_option()
        brd.print_board()
        self.draw_board()
        pygame.display.update()
        
        if game_mode >= 6 and game_mode <= 15:  # Modos IA vs IA
            run_ai_vs_ai_game(self, brd, game_mode)
        else:
            self.play(brd, game_mode)

    def play(self, brd: Board, game_mode: int):
        board = brd.get_board()
        game_over = False
        # Mario !!
        font = pygame.font.SysFont('Connect4-main/fonts/SuperMario256.ttf', 50)
        turns = itertools.cycle([1, 2])
        turn = next(turns)

        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: quit()

                if event.type == pygame.MOUSEMOTION:
                    pygame.draw.rect(self.screen, s.BACKGROUND_COLOR, (0, 0, self.width, self.pixels - 14))
                    posx = event.pos[0]
                    pygame.draw.circle(self.screen, s.PIECES_COLORS[turn], (posx, int(self.pixels / 2) - 7), self.rad)
                pygame.display.update()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if turn == 1 or (turn == 2 and game_mode == 1): 
                        if not game.first_player_move(brd, self, board, turn, event): continue  # verifica se a coluna é valida
                        if game.winning_move(board, turn):
                            game_over = True
                            break
                        turn = next(turns)

                if turn != 1 and game_mode != 1:
                    pygame.time.wait(15)
                    game_over = game.ai_move(brd, self, game_mode, board,turn)  # recebe jogada da IA e retorna se o jogo acabou
                    if game_over: break
                    turn = next(turns)

            if game.is_game_tied(board):
                self.show_draw(font)
                break

        if game.winning_move(board, turn):
            self.show_winner(font, turn)

        pygame.time.wait(10000)

    def draw_menu(self):
        """Desenha o menu e as opções do jogo"""
        
        self.screen.fill(s.BACKGROUND_COLOR)
        font = pygame.font.Font('Connect4-main/fonts/SuperMario256.ttf', 80)

        colors = [s.RED, s.YELLOW, s.BLUE, s.GREEN]
        pos = (560 - font.size("Connect 4")[0] // 2, 230 - font.get_height() // 2)

        self.render_alternating_colors_text("Connect 4", font, colors, pos)

        self.draw_button(self.height / 2, 350, 300, 50, "Single Player")
        self.draw_button(self.height / 2, 450, 300, 50, "Multiplayer")
        self.draw_button(self.height / 2, 550, 300, 50, "PC x PC")

    def render_alternating_colors_text(self, text: str, font, colors, pos):
        """Função para fazer os titulos ficarem igual ao super mario"""
        
        x, y = pos
        outline_color = s.BLACK
        outline_range = range(-3,4)

        for i, char in enumerate(text):
            color = colors[i % len(colors)]

            for ox in outline_range:
                for oy in outline_range:
                    if ox == 0 and oy == 0:
                        continue  # Pula o centro
                    outline_surface = font.render(char, True, outline_color)
                    self.screen.blit(outline_surface, (x + ox, y + oy))

            letter_surface = font.render(char, True, color)
            self.screen.blit(letter_surface, (x, y))
            x += letter_surface.get_width()  # Move x para o próximo caractere
        pygame.display.update()

    def choose_option(self) -> int:
        """Retorna o modo de jogo escolhido"""
        
        while True:
            game_mode = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    if (self.width / 2 - 150) <= mouse_x <= (self.width / 2 + 150) and 350 <= mouse_y <= 400:
                        print("Player vs IA selecionado")
                        self.draw_difficulties()
                        game_mode = 2
                    elif (self.width / 2 - 150) <= mouse_x <= (self.width / 2 + 150) and 450 <= mouse_y <= 500:
                        print("Player vs Player selecionado")
                        game_mode = 1
                    elif (self.width / 2 - 150) <= mouse_x <= (self.width / 2 + 150) and 550 <= mouse_y <= 600:
                        print("IA vs IA selecionado")
                        game_mode = 3

            pygame.display.flip()

            if game_mode == 1:
                return game_mode

            if game_mode == 2:
                game_mode = self.choose_AI_difficulty()
                return game_mode

            if game_mode == 3:
                self.draw_combinations()
                game_mode = self.choose_AI_combination()
                return game_mode

    def draw_combinations(self):
        """Desenha as possíveis combinações entre as IAs"""
        
        self.screen.fill(s.BACKGROUND_COLOR)

        left_x = self.width / 4 - 200
        right_x = self.width * 3 / 4 - 200

        self.draw_button(left_x, 150, 400, 50, "Easy x Easy")  # A* x A*
        self.draw_button(left_x, 250, 400, 50, "Easy x Hard")  # A* x alpha beta
        self.draw_button(left_x, 350, 400, 50, "Medium x Medium")  # mcts x mcts
        self.draw_button(left_x, 450, 400, 50, "Medium x Challenge")  # mcts x decision tree
        self.draw_button(left_x, 550, 400, 50, "Hard x Challenge")  # alpha beta x decision tree

        self.draw_button(right_x, 150, 400, 50, "Easy x Medium")  # A* x mcts
        self.draw_button(right_x, 250, 400, 50, "Easy x Challenge")  # A* x decision tree
        self.draw_button(right_x, 350, 400, 50, "Medium x Hard")  # mcts x alpha beta
        self.draw_button(right_x, 450, 400, 50, "Hard x Hard")  # alpha beta x alpha beta
        self.draw_button(right_x, 550, 400, 50, "Challenge x Challenge")  # decision tree x decision tree

    def choose_AI_combination(self):
        """Retorna qual combinação foi escolhida"""
        
        left_x = self.width / 4 - 200
        right_x = self.width * 3 / 4 - 200
        game_mode = 0

        while True:
            for event in pygame.event.get():
                current_event = event.type
                if current_event == pygame.QUIT:
                    quit()
                elif current_event == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    if left_x <= mouse_x <= left_x + 400:
                        if 150 <= mouse_y <= 200:
                            game_mode = 6
                            print("Easy x Easy")
                        elif 250 <= mouse_y <= 300:
                            game_mode = 7
                            print("Easy x Hard")
                        elif 350 <= mouse_y <= 400:
                            game_mode = 8
                            print("Medium x Medium")
                        elif 450 <= mouse_y <= 500:
                            game_mode = 9
                            print("Medium x Challenge")
                        elif 550 <= mouse_y <= 600:
                            game_mode = 10
                            print("Hard x Challenge")
                    elif right_x <= mouse_x <= right_x + 400:
                        if 150 <= mouse_y <= 200:
                            game_mode = 11
                            print("Easy x Medium")
                        elif 250 <= mouse_y <= 300:
                            game_mode = 12
                            print("Easy x Challenge")
                        elif 350 <= mouse_y <= 400:
                            game_mode = 13
                            print("Medium x Hard")
                        elif 450 <= mouse_y <= 500:
                            game_mode = 14
                            print("Hard x Hard")
                        elif 550 <= mouse_y <= 600:
                            game_mode = 15
                            print("Challenge x Challenge")
            pygame.display.flip()
            if game_mode != 0:
                return game_mode

    def draw_difficulties(self):
        """Desenha as dificuldades para single player"""
        
        self.screen.fill(s.BACKGROUND_COLOR)
        self.draw_button(self.height / 2, 250, 300, 50, "Easy")       # A*
        self.draw_button(self.height / 2, 350, 300, 50, "Medium")     # Monte Carlo (MCTS)
        self.draw_button(self.height / 2, 450, 300, 50, "Hard")       # Alpha Beta
        self.draw_button(self.height / 2, 550, 300, 50, "Challenge")  # Decision Tree

    def choose_AI_difficulty(self):
        game_mode = 0
        while True:
            for event in pygame.event.get():
                current_event = event.type
                if current_event == pygame.QUIT: quit()
                elif current_event == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    if (self.width / 2 - 150) <= mouse_x <= (self.width / 2 + 150) and 250 <= mouse_y <= 300:
                        game_mode = 2
                        print("A*")
                    elif (self.width / 2 - 150) <= mouse_x <= (self.width / 2 + 150) and 350 <= mouse_y <= 400:
                        game_mode = 3
                        print("A* Adversarial")
                    elif (self.width / 2 - 150) <= mouse_x <= (self.width / 2 + 150) and 450 <= mouse_y <= 500:
                        game_mode = 4
                        print("Alpha Beta")
                    elif (self.width / 2 - 150) <= mouse_x <= (self.width / 2 + 150) and 550 <= mouse_y <= 600:
                        game_mode = 5
                        print("MCTS")
                pygame.display.flip()
                if game_mode != 0:
                    return game_mode

    def draw_board(self):
        """Desenha o tabuleiro do jogo"""
        
        self.screen.fill(s.BACKGROUND_COLOR)

        shadow_coordinates = (2 * self.pixels - 10, self.pixels - 10, self.columns * self.pixels + 24, self.rows * self.pixels + 24)
        board_coordinates = (2 * self.pixels - 10, self.pixels - 10, self.columns * self.pixels + 20, self.rows * self.pixels + 20)
        pygame.draw.rect(self.screen, s.GRAY, shadow_coordinates, 0, 30)
        pygame.draw.rect(self.screen, s.BOARD_COLOR, board_coordinates, 0, 30)

        # desenha os espaços vazios:
        for col in range(self.columns):
            for row in range(self.rows):
                center_of_circle = (int((col + 5 / 2) * self.pixels), int((row + 3 / 2) * self.pixels))
                pygame.draw.circle(self.screen, s.BACKGROUND_COLOR, center_of_circle, self.rad)
        pygame.display.update()

    def draw_new_piece(self, row: int, col: int, piece: int):
        """Desenha a nova peça a ser colocada"""
        
        center_of_circle = (int(col * self.pixels + self.pixels / 2), self.height - int(row * self.pixels + self.pixels / 2))
        pygame.draw.circle(self.screen, s.PIECES_COLORS[piece], center_of_circle, self.rad)

    def draw_button(self, x: int, y: int, width: int, height: int, text: str):
        """Desenha os botões de opção"""
        
        pygame.draw.rect(self.screen, s.GRAY, (x, y, width, height), 0, 30)
        font = pygame.font.Font('Connect4-main/fonts/SuperMario256.ttf', 25)
        text_surface = font.render(text, True, s.BLACK)
        text_rect = text_surface.get_rect()
        text_rect.center = (x + width / 2, y + height / 2)
        self.screen.blit(text_surface, text_rect)

    def show_winner(self, font: any, turn: int):
        """Exibe o vencedor"""
        
        font = pygame.font.Font('Connect4-main/fonts/SuperMario256.ttf', 50)
        colors = [s.RED, s.YELLOW, s.BLUE, s.GREEN]
        winner = ("Jogador " + str(turn) + " venceu!")
        pos = (560 - font.size(winner)[0] // 2, 20)
        self.render_alternating_colors_text(winner, font, colors, pos)
        pygame.display.update()

    def show_draw(self, font: any):
        """Exibe mensagem de empate""" 
        
        font = pygame.font.Font('Connect4-main/fonts/SuperMario256.ttf', 50)
        colors = [s.RED, s.YELLOW, s.BLUE, s.GREEN]
        draw_message = "Jogo empatado!"
        pos = (560 - font.size(draw_message)[0] // 2, 20)   
        self.render_alternating_colors_text(draw_message, font, colors, pos)
        pygame.display.update()

    def quit(self):
        pygame.quit()
        sys.exit()
