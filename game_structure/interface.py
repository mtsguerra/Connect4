import pygame
import itertools
import sys
import style as s
import board
import game_engine as game
from dataclasses import dataclass


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

    def starting_game(self, brd: board):
        """Set up the conditions to the game, as choose game_mode and draw the pygame display"""
        pygame.init()
        self.draw_menu()
        game_mode = self.choose_option()
        brd.print_board()
        self.draw_board()
        pygame.display.update()
        self.play(brd, game_mode)

    def play(self, brd: board, game_mode: int):
        board = brd.get_board()
        game_over = False
        font = pygame.font.SysFont('fonts/SuperMario256.ttf', 50)
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

                # jogada do humano:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if turn == 1 or (turn == 2 and game_mode == 1):  # recebe a jogada do humano
                        if not game.human_move(brd, self, board, turn, event): continue  # verifica se a coluna é valida
                        if game.winning_move(board, turn):
                            game_over = True
                            break
                        turn = next(turns)

                # jogada da IA:
                if turn != 1 and game_mode != 1:
                    pygame.time.wait(15)
                    game_over = game.ai_move(brd, self, game_mode, board,
                                             turn)  # recebe jogada da IA e retorna se o jogo acabou,
                    if game_over: break  # seja por vitória ou por empate
                    turn = next(turns)

            if game.is_game_tied(board):
                self.show_draw(font)
                break

        if game.winning_move(board, turn):
            self.show_winner(font, turn)

        # else:
        #     self.show_draw(font)

        pygame.time.wait(10000)

    def draw_menu(self):
        """Use an alternating color to draw it in the SuperMario style, and for the game options board"""
        self.screen.fill(s.BACKGROUND_COLOR)
        font = pygame.font.Font('fonts/SuperMario256.ttf', 80)

        colors = [s.RED, s.YELLOW, s.BLUE, s.GREEN]
        pos = (560 - font.size("Connect 4")[0] // 2, 230 - font.get_height() // 2)

        self.render_alternating_colors_text("Connect 4", font, colors, pos)

        self.draw_button(self.height / 2, 350, 300, 50, "Single Player")
        self.draw_button(self.height / 2, 450, 300, 50, "Multiplayer")
        self.draw_button(self.height / 2, 550, 300, 50, "PC x PC")

    def render_alternating_colors_text(self, text: str, font, colors, pos):
        x, y = pos
        outline_color = s.BLACK
        outline_range = range(-3,4)

        for i, char in enumerate(text):
            color = colors[i % len(colors)]

            for ox in outline_range:
                for oy in outline_range:
                    if ox == 0 and oy == 0:
                        continue  # Skip center

                    outline_surface = font.render(char, True, outline_color)
                    self.screen.blit(outline_surface, (x + ox, y + oy))


            letter_surface = font.render(char, True, color)
            self.screen.blit(letter_surface, (x, y))
            x += letter_surface.get_width()  # Move x for next character
        pygame.display.update()

    def choose_option(self) -> int:
        while True:
            game_mode = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    if (self.width / 2 - 150) <= mouse_x <= (self.width / 2 + 150) and 350 <= mouse_y <= 400:
                        print("Player vs AI selected")
                        self.draw_difficulties()
                        game_mode = 2
                    elif (self.width / 2 - 150) <= mouse_x <= (self.width / 2 + 150) and 450 <= mouse_y <= 500:
                        print("Player vs Player selected")
                        game_mode = 1
                    elif (self.width / 2 - 150) <= mouse_x <= (self.width / 2 + 150) and 550 <= mouse_y <= 600:
                        print("AI vs AI selected")
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
        self.screen.fill(s.BACKGROUND_COLOR)

        left_x = self.width / 4 - 200
        right_x = self.width * 3 / 4 - 200

        self.draw_button(left_x, 150, 400, 50, "Easy x Easy")  # A*
        self.draw_button(left_x, 250, 400, 50, "Easy x Hard")  # A* adversarial
        self.draw_button(left_x, 350, 400, 50, "Medium x Medium")  # Alpha Beta
        self.draw_button(left_x, 450, 400, 50, "Medium x Challenge")  # MCTS
        self.draw_button(left_x, 550, 400, 50, "Hard x Challenge")  # MCTS

        self.draw_button(right_x, 150, 400, 50, "Easy x Medium")  # A*
        self.draw_button(right_x, 250, 400, 50, "Easy x Challenge")  # A* adversarial
        self.draw_button(right_x, 350, 400, 50, "Medium x Hard")  # Alpha Beta
        self.draw_button(right_x, 450, 400, 50, "Hard x Hard")  # MCTS
        self.draw_button(right_x, 550, 400, 50, "Challenge x Challenge")  # MCTS

    def choose_AI_combination(self):
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
                            game_mode = 1
                            print("Easy x Easy")
                        elif 250 <= mouse_y <= 300:
                            game_mode = 2
                            print("Easy x Hard")
                        elif 350 <= mouse_y <= 400:
                            game_mode = 3
                            print("Medium x Medium")
                        elif 450 <= mouse_y <= 500:
                            game_mode = 4
                            print("Medium x Challenge")
                        elif 550 <= mouse_y <= 600:
                            game_mode = 5
                            print("Hard x Challenge")
                    elif right_x <= mouse_x <= right_x + 400:
                        if 150 <= mouse_y <= 200:
                            game_mode = 6
                            print("Easy x Medium")
                        elif 250 <= mouse_y <= 300:
                            game_mode = 7
                            print("Easy x Challenge")
                        elif 350 <= mouse_y <= 400:
                            game_mode = 8
                            print("Medium x Hard")
                        elif 450 <= mouse_y <= 500:
                            game_mode = 9
                            print("Hard x Hard")
                        elif 550 <= mouse_y <= 600:
                            game_mode = 10
                            print("Challenge x Challenge")
            pygame.display.flip()
            if game_mode != 0:
                return game_mode

    def draw_difficulties(self):
        self.screen.fill(s.BACKGROUND_COLOR)
        self.draw_button(self.height / 2, 250, 300, 50, "Easy")  # A*
        self.draw_button(self.height / 2, 350, 300, 50, "Medium")  # A* adversarial
        self.draw_button(self.height / 2, 450, 300, 50, "Hard")  # Alpha Beta
        self.draw_button(self.height / 2, 550, 300, 50, "Challenge")  # MCTS

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
        """Draw pygame board display"""
        self.screen.fill(s.BACKGROUND_COLOR)

        # draw the board and its shadow:
        shadow_coordinates = (2 * self.pixels - 10, self.pixels - 10, self.columns * self.pixels + 24,
                              self.rows * self.pixels + 24)
        board_coordinates = (2 * self.pixels - 10, self.pixels - 10, self.columns * self.pixels + 20,
                             self.rows * self.pixels + 20)
        pygame.draw.rect(self.screen, s.GRAY, shadow_coordinates, 0,
                         30)  # draws the shadow with rounded corners
        pygame.draw.rect(self.screen, s.BOARD_COLOR, board_coordinates, 0, 30)  # draws the board with rounded corners

        # draw the board empty spaces:
        for col in range(self.columns):
            for row in range(self.rows):
                center_of_circle = (int((col + 5 / 2) * self.pixels), int((row + 3 / 2) * self.pixels))
                pygame.draw.circle(self.screen, s.BACKGROUND_COLOR, center_of_circle, self.rad)
        pygame.display.update()

    def draw_new_piece(self, row: int, col: int, piece: int):
        center_of_circle = (int(col * self.pixels + self.pixels / 2),
                            self.height - int(row * self.pixels + self.pixels / 2))
        pygame.draw.circle(self.screen, s.PIECES_COLORS[piece], center_of_circle, self.rad)

    def draw_button(self, x: int, y: int, width: int, height: int, text: str):
        """Draw the option buttons"""
        pygame.draw.rect(self.screen, s.GRAY, (x, y, width, height), 0, 30)
        font = pygame.font.Font('fonts/SuperMario256.ttf', 25)
        text_surface = font.render(text, True, s.BLACK)
        text_rect = text_surface.get_rect()
        text_rect.center = (x + width / 2, y + height / 2)
        self.screen.blit(text_surface, text_rect)

    def show_winner(self, font: any, turn: int):
        """Print the winner"""
        font = pygame.font.Font('fonts/SuperMario256.ttf', 25)

        colors = [s.RED, s.YELLOW, s.BLUE, s.GREEN]
        winner = ("Player " + str(turn) + " wins!")
        pos = (560 - font.size(winner)[0] // 2, 20)


        self.render_alternating_colors_text(winner, font, colors, pos)

        pygame.display.update()

    def show_draw(self, font: any):
        """Print draw game message"""
        label = font.render("Game tied!", True, s.BOARD_COLOR)
        self.screen.blit(label, (400, 15))
        pygame.display.update()

    def quit(self):
        pygame.quit()
        sys.exit()
