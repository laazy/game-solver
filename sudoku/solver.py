from typing import Tuple, List

# puzzle = [
#     #       ||       ||
#     [0, 0, 7, 0, 0, 0, 0, 8, 0],
#     [4, 0, 5, 0, 0, 3, 0, 0, 0],
#     [8, 0, 0, 9, 5, 0, 0, 0, 0],
#     #       ||       ||
#     [0, 0, 0, 0, 0, 8, 1, 0, 0],
#     [9, 0, 0, 3, 6, 7, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 4, 0],
#     #       ||       ||
#     [7, 0, 0, 0, 0, 9, 2, 0, 0],
#     [0, 0, 1, 0, 0, 0, 3, 6, 0],
#     [0, 2, 0, 0, 0, 0, 0, 0, 0]
# ]

puzzle = [
    #       ||       ||
    [0, 1, 0, 3, 8, 6, 9, 0, 2],
    [7, 3, 0, 1, 0, 0, 0, 8, 4],
    [0, 6, 0, 4, 9, 0, 0, 0, 0],
    #       ||       ||
    [8, 5, 6, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 8, 0, 5, 7, 0, 0],
    [0, 0, 0, 0, 0, 0, 2, 5, 8],
    #       ||       ||
    [0, 0, 0, 0, 7, 8, 0, 2, 0],
    [6, 9, 0, 0, 0, 4, 0, 3, 7],
    [1, 0, 7, 2, 6, 3, 0, 9, 0]
]


class Cell:
    def __init__(self, num: int, pos: Tuple[int, int]):
        self.pos = pos
        self.hint = []
        self.num = num
        if num == 0:
            self.hint = list(range(1, 10))

    def __str__(self):
        return '\n'.join(self.get_expr())

    def get_expr(self):
        if self.num != 0:
            return ['      ', f'   {self.num}  ', '      ']
        s = ''
        for i in range(1, 10):
            s += f' {i}' if i in self.hint else ' 0'
        strip = len(s) // 3
        return [s[i: i + strip] for i in range(0, len(s), strip)]

    def update(self):
        if len(self.hint) == 1:
            self.num = self.hint[0]
            self.hint = []

    def remove(self, n: int):
        if n in self.hint:
            self.hint.remove(n)

    def __hash__(self):
        return id(self)


Cells = List[Cell]
Matrix = List[Cells]


class Solver:
    def __init__(self):
        self.board: Matrix = None
        self.rows: Matrix = None
        self.cols: Matrix = None
        self.tiny: Matrix = None

    def solve(self):
        pass

    @staticmethod
    def check_puzzle(puzzle) -> Tuple[bool, str]:
        if len(puzzle) != 9 or not all((len(i) == 9 for i in puzzle)):
            return False, "Must be 9*9"
        if not all(type(i) is int and i >= 0 and i <= 9 for j in puzzle for i in j):
            return False, "Must input integer"
        return True, None

    def load_puzzle(self, puzzle):
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
            self.board[i // 3 * 3 + j // 3][i % 3 * 3 + j % 3] for j in range(9)]
            for i in range(9)]

    def get_row(self, c: Cell):
        return self.rows[c.pos[0]]

    def get_col(self, c: Cell):
        return self.cols[c.pos[1]]

    def get_small(self, c: Cell):
        x, y = c.pos
        return self.tiny[x // 3 * 3 + y // 3]

    def get_neighbor(self, c: Cell) -> List[Cell]:
        return set().union(self.get_col(c), self.get_row(c), self.get_small(c))

    def remove_note(self, c: Cell) -> None:
        if c.num != 0:
            for k in self.get_neighbor(c):
                k.remove(c.num)
                k.update()

    def set_only(self, cells: List[Cell]) -> None:
        s = {i: list(filter(lambda c: i in c.hint, cells)) for i in range(9)}
        for k, v in s.items():
            if len(v) == 1:
                v[0].num = k

    def iter(self):
        for i in self.board:
            for j in i:
                self.remove_note(j)

        for i in range(9):
            self.set_only(self.rows[i])
            self.set_only(self.cols[i])
            self.set_only(self.tiny[i])

            

    def check(self):
        pass

    def __str__(self):
        ans = []
        for i in range(9):
            row = [j.get_expr() for j in self.board[i]]
            t = ['  '.join(i) for i in zip(*row)] + ['']
            ans += t
        return "\n".join(ans)


if __name__ == "__main__":
    solver = Solver()
    solver.load_puzzle(puzzle)
    while True:
        print(solver)
        solver.iter()
        input()
