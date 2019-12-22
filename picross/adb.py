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

ADB = "adb"


def take_screenshot(img_path):
    os.system(f"{ADB} shell screencap -p /sdcard/game-solver-image.png")
    os.system(f"{ADB} pull /sdcard/game-solver-image.png {img_path}")


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


def need_first_touch():
    return True


def simulate_touch(board):
    if need_first_touch():
        os.system(f"{ADB} shell input tap 400 800")
    dim = len(board)
    for i, row in enumerate(board):
        grouped_row = [(x, len(list(g))) for x, g in groupby(row)]
        print(grouped_row)
        col_begin = 0
        for value, cnt in grouped_row:
            if value == 1:
                swipe(dim, i, col_begin, cnt)
            col_begin += cnt


if __name__ == '__main__':
    pass
