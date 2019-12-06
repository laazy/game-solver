from typing import List

Matrix = List[List]

puzzle = [
    [[2], [4], [4], [4], [2]],
    [[1, 1], [5], [5], [3], [1]]
]

answer = [
    [0, 1, 0, 1, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0]
]


def generate_all_possible(puzzle: Matrix) -> Matrix:
    pass


def ignore_impossible(possibility: Matrix, board: Matrix) -> Matrix:
    pass


def count_absolute_answer(possibility: Matrix) -> Matrix:
    pass


def matched(puzzle: Matrix, board: Matrix) -> bool:
    return True


def main():
    dim = len(puzzle[0])
    board = [[0] * dim for _ in range(dim)]
    possibility = generate_all_possible(puzzle)
    while not matched(puzzle, board):
        absolute = ignore_impossible(possibility, board)
        board = count_absolute_answer(absolute, board)
