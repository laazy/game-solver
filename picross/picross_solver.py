from typing import List
from itertools import cycle
import sys
import time

Matrix = List[List[int]]

ROW = 0
COL = 1


class Solver:
    def __init__(self):
        self.puzzle_loaded = False
        self.row = 0
        self.col = 0
        self.board: List[List[int]] = []
        self.row_info = []
        self.col_info = []

    @staticmethod
    def str_to_hint_matrix(s: str) -> List[List[int]]:
        return [[int(j) for j in i.split()] for i in s.split(',')]

    def print_board(self):
        for rows in self.board:
            for item in rows:
                print(f"{item if item == 1 else 0 if item == -1 else 'N':>3}", end="")
            print()

    def load_puzzle(self, file_path: str):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        self.row, self.col = (int(i) for i in lines[0].strip().split())
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
        return cycle(res)

    def solve(self):
        row_p13s, col_p13s = [None] * self.col, [None] * self.row
        round_orders = self.cal_orders()
        for item in round_orders:
            index = item["index"]
            if item["type"] == ROW:
                line = self.board[index]
                info = self.row_info[index]
                p13s = row_p13s
            else:
                line = [i[index] for i in self.board]
                info = self.col_info[index]
                p13s = col_p13s

            if 0 not in line:
                continue

            p13s[index] = p13s[index] or self.gen_line(info, line)
            p13s[index] = self.ignore_impossible(p13s[index], line)
            absolute_answer = self.count_absolute_answer(p13s[index])

            if item["type"] == ROW:
                self.board[index] = absolute_answer
            else:
                for i in range(self.row):
                    self.board[i][index] = absolute_answer[i]
            if self.matched():
                break

    def gen_line(self, info: List, line: List) -> Matrix:
        length = len(line)
        if not info:
            return [[-1] * length] if 1 not in line else []
        ele = info[0]
        ans: Matrix = []
        for i in range(length - ele + 1):
            if sum(info[1:]) + len(info[1:]) - 1 > length - ele - i:
                break
            header = [-1] * i + [1] * ele + ([-1] if i + ele < length else [])
            if self.no_conflict(header, line[:len(header)]):
                next_ans = self.gen_line(info[1:], line[len(header):])
                ans.extend([header + j for j in next_ans])
        return ans

    @staticmethod
    def no_conflict(pos: List, row: List) -> bool:
        return all(i*j != -1 for i, j in zip(pos, row))

    def ignore_impossible(self, possibility: Matrix, row: List) -> Matrix:
        return [i for i in possibility if self.no_conflict(i, row)]

    @staticmethod
    def count_absolute_answer(possibility: Matrix) -> List:
        count_table = {-len(possibility): -1, len(possibility): 1}
        return [count_table.get(sum(i), 0) for i in zip(*possibility)]

    def matched(self) -> bool:
        return all([all(i) for i in self.board])


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
