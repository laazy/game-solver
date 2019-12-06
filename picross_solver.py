from typing import List

Puzzle = List[Matrix]
Matrix = List[List]

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


def generate_all_possible(row_puzzle: Matrix) -> Matrix:
    pass


def ignore_impossible(possibility: Matrix, row: List) -> Matrix:
    pass


def count_absolute_answer(possibility: Matrix) -> List:
    pass


def matched(puzzle: Puzzle, board: Matrix) -> bool:
    return True


def transpose(board: Matrix) -> None:
    pass


def main():
    dim = len(puzzle[0])
    board = [[0] * dim for _ in range(dim)]
    # possibilities = [[0] * dim for i in range(2)]
    row_possibilities = generate_all_possible(puzzle[1])
    col_possibilities = generate_all_possible(puzzle[0])
    while not matched(puzzle, board):
        # compute row
        for i in range(dim):
            row_possibilities[i] = ignore_impossible(
                row_possibilities[i], board[i])
            board[i] = count_absolute_answer(row_possibilities[i])
        # compute col in next iteration
        transpose(board)
