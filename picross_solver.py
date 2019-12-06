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


def generate_all_possible(row_puzzle: Matrix) -> List[Matrix]:
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
