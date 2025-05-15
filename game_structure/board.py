import numpy as np
from dataclasses import dataclass, field
import style as s

@dataclass
class Board:
    rows: int = s.ROWS
    columns: int = s.COLUMNS
    board: np.ndarray = field(default_factory=lambda: np.zeros((s.ROWS, s.COLUMNS)))

    def get_board(self):
        return self.board

    def print_board(self):
        print(np.flip(self.board, 0), "\n")
