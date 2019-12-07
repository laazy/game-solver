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
    if not line:
        return [[0] * length]
    ele = line[0]
    ans = []
    for i in range(length - ele + 1):
        if sum(line[1:]) + len(line[1:]) - 1 + ele > length:
            continue
        next_ans = gen_line(line[1:], length - ele - i)
        ans.extend([[0] * i + [1] * ele + j for j in next_ans])
    return ans


def generate_all_possible(row_puzzle: Matrix) -> List[Matrix]:
    return [gen_line(i, dim) for i in row_puzzle]


def ignore_impossible(possibility: Matrix, row: List) -> Matrix:
    pass


def count_absolute_answer(possibility: Matrix) -> List:
    pass


def matched(puzzle: Puzzle, board: Matrix) -> bool:
    return True


def transpose(board: Matrix) -> None:
    for i in range(dim):
        for j in range(dim):
            board[i][j],  board[j][i] = board[j][i], board[i][j]


def main():
    board = [[0] * dim for _ in range(dim)]
    row_possibilities = generate_all_possible(puzzle[1])
    col_possibilities = generate_all_possible(puzzle[0])
    possibilities = row_possibilities
    while not matched(puzzle, board):
        # compute row
        for i in range(dim):
            possibilities[i] = ignore_impossible(
                possibilities[i], board[i])
            board[i] = count_absolute_answer(possibilities[i])
        # compute col in next iteration
        transpose(board)
        possibilities = row_possibilities if possibilities is col_possibilities else col_possibilities


if __name__ == "__main__":
    row_possibilities = generate_all_possible(puzzle[1])
    print(row_possibilities)
