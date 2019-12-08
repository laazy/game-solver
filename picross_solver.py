from typing import List
from itertools import cycle
import sys
import time

Matrix = List[List[int]]
Puzzle = List[Matrix]

ROW = 0
COL = 1


class Solver:
    def __init__(self):
        self.puzzle_loaded = False
        self.row = 0
        self.col = 0
        self.board: List[List(int)] = None
        self.row_info = []
        self.col_info = []

    @staticmethod
    def str_to_hint_matrix(s: str) -> List[int]:
        return [[int(j) for j in i.split()] for i in s.split(',')]

    def print_board(self):
        for rows in self.board:
            for item in rows:
                print(f"{item if item == 1 else 0:3}", end="")
            print()

    def load_puzzle(self, file_path: str):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        self.row, self.col = (int(i) for i in lines[0].strip().split())
        self.dim = self.row
        self.board = [[0] * self.col for _ in range(self.row)]
        self.col_info = self.str_to_hint_matrix(lines[1])
        self.row_info = self.str_to_hint_matrix(lines[2])
        assert self.row == len(self.row_info)
        assert self.col == len(self.col_info)

    def cal_orders(self):
        res = []
        for index, row in enumerate(self.row_info):
            res.append({
                "type": ROW,
                "index": index,
                "score": sum(row) + len(row) - 1
            })
        for index, col in enumerate(self.col_info):
            res.append({
                "type": COL,
                "index": index,
                "score": sum(col) + len(col) - 1
            })
        res.sort(key=lambda item: item["score"], reverse=True)
        return res

    def solve(self):
        row_possibilities = [None] * self.col
        col_possibilities = [None] * self.row

        round_orders = self.cal_orders()
        while not self.matched():
            print("new round")
            cnt = 0
            for item in round_orders:
                cnt += 1
                print(cnt)
                index = item["index"]
                if item["type"] == ROW:
                    line = self.board[index]
                    info = self.row_info[index]
                else:
                    line = [i[index] for i in self.board]
                    info = self.col_info[index]

                if 0 not in line:
                    continue

                if item["type"] == ROW:
                    if row_possibilities[index] is None:
                        row_possibilities[index] = self.gen_line(info, line)
                    row_possibilities[index] = self.ignore_impossible(
                        row_possibilities[index], line)
                    possibilities = row_possibilities[index]
                else:
                    if col_possibilities[index] is None:
                        col_possibilities[index] = self.gen_line(info, line)
                    col_possibilities[index] = self.ignore_impossible(
                        col_possibilities[index], line)
                    possibilities = col_possibilities[index]

                absolute_answer = self.count_absolute_answer(possibilities)
                if item["type"] == ROW:
                    self.board[index] = absolute_answer
                else:
                    for i in range(self.row):
                        self.board[i][index] = absolute_answer[i]

    def gen_line(self, info: List, line: List) -> Matrix:
        length = len(line)

        def _gen(i):
            ans = [-1] * i + [1] * ele
            if i + ele < length:
                ans.append(-1)
            return ans

        if not info:
            return [[-1] * length]
        ele = info[0]
        ans = []
        for i in range(length - ele + 1):
            if sum(info[1:]) + len(info[1:]) - 1 > length - ele - i:
                break
            header = _gen(i)
            if not self.no_conflict(header, line[:len(header)]):
                continue
            next_ans = self.gen_line(info[1:], line[len(header):])
            ans.extend([header + j for j in next_ans])
        return ans

    def no_conflict(self, pos: List, row: List) -> bool:
        for i in range(len(pos)):
            if row[i] != 0 and pos[i] != row[i]:
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
        count_table = {-len(possibility): -1, len(possibility): 1}
        for i in zip(*possibility):
            ans.append(count_table.get(sum(i), 0))
        return ans

    def matched(self) -> bool:
        for i in self.board:
            if 0 in i:
                return False
        return True

    def transpose(self) -> None:
        for i in range(self.dim):
            for j in range(i, self.dim):
                self.board[i][j],  self.board[j][i] = self.board[j][i], self.board[i][j]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("You must specify file name")
        exit(-1)
    solver = Solver()
    solver.load_puzzle(sys.argv[1])
    _start = time.time()
    solver.solve()
    _end = time.time()
    solver.print_board()
    print(f"time spent: {_end - _start}")
