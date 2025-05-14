from game_structure.board import Board
from game_structure.interface import Interface


def main() -> None:
    board = Board()
    interface = Interface()
    interface.start_engine(board)


if __name__ == "__main__":
    main()