from typing import Tuple, List

puzzle = [
    #       ||       ||
    [0, 0, 4, 0, 0, 7, 0, 6, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 0, 9, 3, 0],
    #       ||       ||
    [0, 0, 0, 0, 0, 5, 0, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 4, 0],
    [0, 0, 0, 0, 3, 2, 5, 7, 0],
    #       ||       ||
    [5, 0, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 4, 0, 0, 0, 1, 0],
    [9, 0, 0, 6, 0, 0, 3, 0, 0]
]

# define two range
INDEX_RANGE = range(9)
NOTE_RANGE = range(1, 10)


class Cell:
    '''
    This class represent the small box in game board. The whole
    boars consist of 9*9 `Cell`
    To avoid misunderstanding, cell below represent number in cell,
    note below represent note in cell.
    '''

    def __init__(self, num: int, pos: Tuple[int, int]):
        # init cell with num.
        self.pos = pos
        self.note = []
        self.num = num
        # if num is 0, fill note with all possibilities
        if num == 0:
            self.note = list(NOTE_RANGE)

    def __str__(self):
        return '\n'.join(self.get_expr())

    def get_expr(self):
        '''
        Get output of a cell. 
        Filled:   |   Unfilled:
                  |   1 2 3
        9         |   4 0 6
                  |   0 0 0 
        '''
        if self.num != 0:
            return ['      ', f'   {self.num}  ', '      ']
        s = ''
        for i in NOTE_RANGE:
            s += f' {i}' if i in self.note else ' 0'
        strip = len(s) // 3
        return [s[i: i + strip] for i in range(0, len(s), strip)]

    def update(self, num=0) -> bool:
        # update cell by note or input, return if it is filled
        if len(self.note) == 1:
            self.num = self.note.pop()
            return True
        elif num != 0:
            self.num = num
            self.note = []
            return True
        return False

    def remove(self, n: int):
        # remove note
        if n in self.note:
            self.note.remove(n)

    def __hash__(self):
        return id(self)


# define types
Cells = List[Cell]
Matrix = List[Cells]


class Solver:
    '''
    This class if solver for sudoku
    '''

    def __init__(self):
        # define while board, row set, col set, tiny board set.
        self.board: Matrix = None
        self.rows: Matrix = None
        self.cols: Matrix = None
        self.tiny: Matrix = None

    def solve(self):
        while not self.check():
            self.iter()

    @staticmethod
    def check_puzzle(puzzle) -> Tuple[bool, str]:
        # check the validation of puzzle
        if len(puzzle) != 9 or not all((len(i) == 9 for i in puzzle)):
            return False, "Must be 9*9"
        if not all(type(i) is int and i >= 0 and i <= 9 for j in puzzle for i in j):
            return False, "Must input integer"
        return True, None

    def load_puzzle(self, puzzle):
        # load puzzle from input
        v, m = self.check_puzzle(puzzle)
        if not v:
            print(m)
            return
        self.board = [[
            Cell(j, (ii, jj)) for jj, j in enumerate(i)
        ] for ii, i in enumerate(puzzle)]
        self.rows = self.board
        self.cols = list(zip(*self.rows))
        self.tiny = [[
            self.board[i // 3 * 3 + j // 3][i % 3 * 3 + j % 3] for j in INDEX_RANGE]
            for i in INDEX_RANGE]

    def get_row(self, c: Cell) -> Cells:
        # get the row where this cell is
        return self.rows[c.pos[0]]

    def get_col(self, c: Cell) -> Cells:
        # get the col where this cell is
        return self.cols[c.pos[1]]

    def get_tiny(self, c: Cell) -> Cells:
        # get the tiny board where this cell is
        x, y = c.pos
        return self.tiny[x // 3 * 3 + y // 3]

    def get_neighbor(self, c: Cell) -> List[Cell]:
        return set().union(self.get_col(c), self.get_row(c), self.get_tiny(c))

    def remove_note(self, c: Cell) -> None:
        # remove note according to this cell
        if c.num != 0:
            for k in self.get_neighbor(c):
                k.remove(c.num)

    def set_only(self, cells: List[Cell]) -> None:
        '''
        If a row/col/tiny board only have 1 cell have some number in note,
        this cell must be that number.
        '''
        for note in NOTE_RANGE:
            s = list(filter(lambda c: note in c.note, cells))
            if len(s) == 1:
                s[0].update(note)
                self.remove_note(s[0])

    def exclude_other(self, num: int) -> None:
        '''
        If some tiny board has a number in note aligned,
        other note in the line could not have this number.
        '''
        cells = self.tiny[num]
        for note in NOTE_RANGE:
            temp = list(filter(lambda c: note in c.note, cells))
            if len({i.pos[0] for i in temp}) == 1:
                for i in self.get_row(temp[0]):
                    if i not in temp:
                        i.remove(note)
            if len({i.pos[1] for i in temp}) == 1:
                for i in self.get_col(temp[0]):
                    if i not in temp:
                        i.remove(note)

    def iter(self):
        # Iterate the solver process.
        for i in self.board:
            for j in i:
                self.remove_note(j)
                if j.update():
                    self.remove_note(j)

        self.check_invariants()

        for i in INDEX_RANGE:
            self.set_only(self.rows[i])
            self.set_only(self.cols[i])
            self.set_only(self.tiny[i])

        self.check_invariants()

        for i in INDEX_RANGE:
            self.exclude_other(i)

        self.check_invariants()

    def check_invariants(self):
        '''
        Check invariant, affirm the board will not break invariants as below.
        1. A number will not show up twice in cells of rows/col tiny board.
        2. A number will not show up in cells meantime as the notes.
        '''
        ans = True
        for index, i in enumerate([*self.rows, *self.cols, *self.tiny]):
            s = {j.num for j in i}
            s.intersection_update(NOTE_RANGE)
            if len(s) != len([j.num for j in i if j.num != 0]):
                print(f"found same number between cells in {index}")
            if any([True if s.intersection(j.note) else False for j in i]):
                print(f"found same number between cell ans notes in {index}")
                ans = False
        return ans

    def check(self):
        # check whether all filled correctly.
        for i in [*self.rows, *self.cols, *self.tiny]:
            s = {j.num for j in i}
            if s.symmetric_difference(NOTE_RANGE):
                return False
        return True

    def __str__(self):
        ans = []
        for i in INDEX_RANGE:
            row = [j.get_expr() for j in self.board[i]]
            t = ['  '.join(i) for i in zip(*row)] + ['']
            ans += t
        return "\n".join(ans)


if __name__ == "__main__":
    solver = Solver()
    solver.load_puzzle(puzzle)
    solver.solve()
    print(solver)
