import math

class GameMeta:
    PLAYERS = {'none': 0, 'one': 1, 'two': 2}
    OUTCOMES = {'none': 0, 'one': 1, 'two': 2, 'draw': 3}
    INF = float('inf')
    ROWS: int = 6
    COLS: int = 7


class MCTSMeta:
    EXPLORATION: float = math.sqrt(2)
