import numpy as np
from dataclasses import dataclass

import style


@dataclass
class Board:
    rows: int = style.ROWS
    columns: int = style.COLUMNS
    board: np.ndarray = np.zeros((rows, columns))

    def get_board(self) -> np.ndarray:
        return self.board

    def print_board(self) -> None:
        print(np.flip(self.board, 0), "\n")