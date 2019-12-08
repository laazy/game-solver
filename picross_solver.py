from typing import List

Matrix = List[List]
Puzzle = List[Matrix]


class Solver:
    def __init__(self):
        self.puzzle_loaded = False
        self.row = 0
        self.col = 0
        self.board: List[List(int)] = None
        self.row_info = []
        self.col_info = []

    @staticmethod
    def str_list_to_int_list(str_list: List[str]) -> List[int]:
        return list(map(lambda item: int(item), str_list))

    def print_board(self):
        for rows in self.board:
            for item in rows:
                if item is None:
                    print("N  ", end="")
                else:
                    print(f"{item:3}", end="")
            print()

    def load_puzzle(self, file_path: str):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        size_line = lines[0].split()
        self.row = int(size_line[0])
        self.col = int(size_line[1])
        self.dim = self.row
        self.board = [[None] * self.col for _ in range(self.row)]
        row_info_line = lines[1].split(',')
        self.row_info = list(
            map(lambda item: self.str_list_to_int_list(item.split()), row_info_line))
        col_info_line = lines[2].split(',')
        self.col_info = list(
            map(lambda item: self.str_list_to_int_list(item.split()), col_info_line))
        assert self.row == len(self.row_info)
        assert self.col == len(self.col_info)
        # self.print_board()
        # print(self.row_info)
        # print(self.col_info)

    def solve(self):
        row_possibilities = [self.gen_line(i, self.dim) for i in self.row_info]
        col_possibilities = [self.gen_line(i, self.dim) for i in self.col_info]
        possibilities = row_possibilities
        while not self.matched():
            # compute row
            for i in range(self.dim):
                possibilities[i] = self.ignore_impossible(
                    possibilities[i], self.board[i])
                self.board[i] = self.count_absolute_answer(possibilities[i])
            # compute col in next iteration
            self.transpose()
            possibilities = row_possibilities if possibilities is col_possibilities else col_possibilities
        if possibilities is col_possibilities:
            self.transpose(board)

    def output(self):
        self.print_board()

    def gen_line(self, line: List, length: int) -> Matrix:
        def _gen(i):
            ans = [0] * i + [1] * ele
            if i + ele < length:
                ans.append(0)
            return ans

        if not line:
            return [[0] * length]
        ele = line[0]
        ans = []
        for i in range(length - ele + 1):
            if sum(line[1:]) + len(line[1:]) - 1 > length - ele - i:
                break
            next_ans = self.gen_line(line[1:], length - ele - i - 1)
            ans.extend([_gen(i) + j for j in next_ans])
        return ans

    def no_conflict(self, pos: List, row: List) -> bool:
        for i in range(self.dim):
            if row[i] is not None and pos[i] != row[i]:
                return False
        return True

    def ignore_impossible(self, possibility: Matrix, row: List) -> Matrix:
        ans = []
        for i in possibility:
            if self.no_conflict(i, row):
                ans.append(i)
        return ans

    def count_absolute_answer(self, possibility: Matrix) -> List:
        ans = []
        count_table = {0: 0, len(possibility): 1}
        for i in zip(*possibility):
            ans.append(count_table.get(sum(i), None))
        return ans

    def matched(self) -> bool:
        for i in self.board:
            if None in i:
                return False
        return True

    def transpose(self) -> None:
        for i in range(self.dim):
            for j in range(i, self.dim):
                self.board[i][j],  self.board[j][i] = self.board[j][i], self.board[i][j]


if __name__ == "__main__":
    solver = Solver()
    solver.load_puzzle("6739.txt")
    solver.solve()
    solver.output()
