from picross_solver import Solver
from itertools import groupby
import os


CONFIG = {
    "5": {
        "base_x": 200,
        "base_y": 485,
        "grid_size": 176
    },
    "25": {
        "base_x": 218,
        "base_y": 626,
        "grid_size": 34
    },
    "30": {
        "base_x": 227,
        "base_y": 636,
        "grid_size": 28
    }
}

ADB = r"D:\Util\platform-tools\adb.exe"
DST_IMAGE = r"D:\projects\gamer-solver\image.png"


def get_image():
    os.system(f"{ADB} shell screencap -p /sdcard/game-solver-image.png")
    os.system(f"{ADB} pull /sdcard/game-solver-image.png {DST_IMAGE}")


def swipe(size, row, col_begin, length):
    config = CONFIG[str(size)]
    x, y = config["base_x"], config["base_y"]
    grid_size = config["grid_size"]
    x += col_begin * grid_size + grid_size // 2
    y += row * grid_size + grid_size // 2

    if length == 1:
        os.system(f"{ADB} shell input tap {x} {y}")
    else:
        os.system(
            f"{ADB} shell input swipe {x} {y} {x + (length - 1) * grid_size} {y}")


if __name__ == '__main__':
    # solver = Solver()
    # solver.load_puzzle("picross/puzzles/simple.txt")
    # solver.solve()
    # board = solver.board
    # for i, row in enumerate(board):
    #     grouped_row = [(x, len(list(g))) for x, g in groupby(row)]
    #     print(grouped_row)
    #     col_begin = 0
    #     for value, cnt in grouped_row:
    #         if value == 1:
    #             swipe(solver.row, i, col_begin, cnt)
    #         col_begin += cnt
    get_image()
    pass
