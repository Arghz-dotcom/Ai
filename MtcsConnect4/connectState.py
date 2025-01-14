from copy import deepcopy
from meta import GameMeta


class ConnectState():
    def __init__(self):
        self.board: list[list[int]] = [[0] * GameMeta.COLS for _ in range(GameMeta.ROWS)]
        """
        Board:
        0 |
          |
          |
          |
          |
        5 |
          --------
          0      6

          6 rows, 7 columns
          Start to play at row 6
        """
        self.to_play: int = GameMeta.PLAYERS['one']
        """ Player to play in this state """
        self.height: list[int] = [GameMeta.ROWS - 1] * GameMeta.COLS
        """ Available row for a col """
        self.last_played: list[int] = []
        """ last played [height, col] """

    def __has_legal_moves(self) -> bool:
        """ Is it still possible to play in the board """
        return any(self.board[0][col] == 0 for col in range(GameMeta.COLS))

    def __check_win(self) -> int:
        """ Return player who won or draw game """

        if len(self.last_played) > 0 and self.__check_win_from(self.last_played[0], self.last_played[1]):
            return self.board[self.last_played[0]][self.last_played[1]]
        return 0 # draw game

    def __check_win_from(self, row: int, col: int) -> bool:
        player: int = self.board[row][col]
        """
        Last played action is at (row, col)
        Check surrounding 7x7 grid for a win
        """

        consecutive: int = 1
        # Check horizontal
        tmprow = row
        while tmprow + 1 < GameMeta.ROWS and self.board[tmprow + 1][col] == player:
            consecutive += 1
            tmprow += 1
        tmprow = row
        while tmprow - 1 >= 0 and self.board[tmprow - 1][col] == player:
            consecutive += 1
            tmprow -= 1

        if consecutive >= 4:
            return True

        # Check vertical
        consecutive = 1
        tmpcol = col
        while tmpcol + 1 < GameMeta.COLS and self.board[row][tmpcol + 1] == player:
            consecutive += 1
            tmpcol += 1
        tmpcol = col
        while tmpcol - 1 >= 0 and self.board[row][tmpcol - 1] == player:
            consecutive += 1
            tmpcol -= 1

        if consecutive >= 4:
            return True

        # Check diagonal
        consecutive = 1
        tmprow = row
        tmpcol = col
        while tmprow + 1 < GameMeta.ROWS and tmpcol + 1 < GameMeta.COLS and self.board[tmprow + 1][tmpcol + 1] == player:
            consecutive += 1
            tmprow += 1
            tmpcol += 1
        tmprow = row
        tmpcol = col
        while tmprow - 1 >= 0 and tmpcol - 1 >= 0 and self.board[tmprow - 1][tmpcol - 1] == player:
            consecutive += 1
            tmprow -= 1
            tmpcol -= 1

        if consecutive >= 4:
            return True

        # Check anti-diagonal
        consecutive = 1
        tmprow = row
        tmpcol = col
        while tmprow + 1 < GameMeta.ROWS and tmpcol - 1 >= 0 and self.board[tmprow + 1][tmpcol - 1] == player:
            consecutive += 1
            tmprow += 1
            tmpcol -= 1
        tmprow = row
        tmpcol = col
        while tmprow - 1 >= 0 and tmpcol + 1 < GameMeta.COLS and self.board[tmprow - 1][tmpcol + 1] == player:
            consecutive += 1
            tmprow -= 1
            tmpcol += 1

        if consecutive >= 4:
            return True

        return False
    
    #def get_board(self):
    #    return deepcopy(self.board)

    def move(self, col: int):
        self.board[self.height[col]][col] = self.to_play
        self.last_played = [self.height[col], col]
        self.height[col] -= 1
        self.to_play = GameMeta.PLAYERS['two'] if self.to_play == GameMeta.PLAYERS['one'] else GameMeta.PLAYERS['one'] # alternate players

    def get_legal_moves(self) -> list[int]:
        """ List of columns that can be played """
        return [col for col in range(GameMeta.COLS) if self.board[0][col] == 0]

    def game_over(self) -> bool:
        return self.__check_win() or not self.__has_legal_moves()

    def get_outcome(self) -> int:
        player_winner = self.__check_win()
        if not self.__has_legal_moves() and player_winner == 0:
            return GameMeta.OUTCOMES['draw']

        return GameMeta.OUTCOMES['one'] if player_winner == GameMeta.PLAYERS['one'] else GameMeta.OUTCOMES['two']

    def print(self):
        print('=============================')

        for row in range(GameMeta.ROWS):
            for col in range(GameMeta.COLS):
                print('| {} '.format('X' if self.board[row][col] == 1 else 'O' if self.board[row][col] == 2 else ' '), end='')
            print('|')

        print('=============================')
