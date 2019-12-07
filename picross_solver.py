from typing import List

Matrix = List[List]
Puzzle = List[Matrix]

puzzle = [
    [[2], [4], [4], [4], [2]],  # top hint
    [[1, 1], [5], [5], [3], [1]]  # left hint
]

answer = [
    [0, 1, 0, 1, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0]
]

dim = len(puzzle[0])


def gen_line(line: List, length: int) -> Matrix:
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
        next_ans = gen_line(line[1:], length - ele - i - 1)
        ans.extend([_gen(i) + j for j in next_ans])
    return ans


def generate_all_possible(row_puzzle: Matrix) -> List[Matrix]:
    return [gen_line(i, dim) for i in row_puzzle]


def no_conflict(pos: List, row: List) -> bool:
    for i in range(dim):
        if row[i] is not None and pos[i] != row[i]:
            return False
    return True


def ignore_impossible(possibility: Matrix, row: List) -> Matrix:
    ans = []
    for i in possibility:
        if no_conflict(i, row):
            ans.append(i)
    return ans


def count_absolute_answer(possibility: Matrix) -> List:
    ans = []
    count_table = {0: 0, len(possibility): 1}
    for i in zip(*possibility):
        ans.append(count_table.get(sum(i), None))
    return ans


def matched(board: Matrix) -> bool:
    for i in board:
        if None in i:
            return False
    return True


def transpose(board: Matrix) -> None:
    for i in range(dim):
        for j in range(i, dim):
            board[i][j],  board[j][i] = board[j][i], board[i][j]


def main():
    board = [[None] * dim for _ in range(dim)]
    row_possibilities = generate_all_possible(puzzle[1])
    col_possibilities = generate_all_possible(puzzle[0])
    possibilities = row_possibilities
    while not matched(board):
        # compute row
        for i in range(dim):
            possibilities[i] = ignore_impossible(
                possibilities[i], board[i])
            board[i] = count_absolute_answer(possibilities[i])
        # compute col in next iteration
        transpose(board)
        possibilities = row_possibilities if possibilities is col_possibilities else col_possibilities
    return board


if __name__ == "__main__":
    board = main()
    for i in board:
        print(i)
