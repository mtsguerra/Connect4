from game_structure.Board import Board
from game_structure.interface import Interface


def main() -> None:
    board = Board()
    interface = Interface()
    interface.starting_game(board)


if __name__ == "__main__":
    main()
