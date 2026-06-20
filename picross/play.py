"""One-command picross autoplay: solve the current puzzle and open the next.

    ADB=/path/to/adb python3 play.py [count]

Pipeline per level: screenshot -> detect grid (auto size) -> OCR clues ->
consistency check -> solve -> activate fill mode -> swipe-fill -> verify and
re-tap dropped cells until the win screen -> navigate to the next level.
"""
import sys
import time

import auto
import clue_ocr

YELLOW_REGION = (2300, 2950)   # y-band of the control buttons


def _yellow_frac(img_path):
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(img_path).convert("RGB")).astype(np.int32)
    reg = a[YELLOW_REGION[0]:YELLOW_REGION[1], :, :]
    R, G, B = reg[:, :, 0], reg[:, :, 1], reg[:, :, 2]
    return float(((R > 180) & (G > 140) & (B < 110)).mean())


def in_puzzle(img_path):
    """True while the puzzle (with its yellow controls) is on screen."""
    return _yellow_frac(img_path) > 0.02


def read_solvable():
    """Screenshot, detect the grid and solve the clues. Returns (g, board) or
    None. We never auto-RESET an unreadable board -- the in-game RESET wipes
    saved progress -- so a board that can't be read is simply reported."""
    img = auto.screenshot("/tmp/play.png")
    try:
        g = auto.detect_grid(img, n=None)
        _, _, board = clue_ocr.solvable_clues(img, g)
        return g, board
    except (AssertionError, RuntimeError) as e:
        print(f"  ! {e}")
        return None


def _won(tries=6, gap=0.35):
    # check immediately (swipe-fill usually completes in one pass), then poll
    # briefly for the victory screen instead of always waiting a fixed time.
    for _ in range(tries):
        if not in_puzzle(auto.screenshot("/tmp/play.png")):
            return True
        time.sleep(gap)
    return False


def solve_current():
    """Solve the puzzle on screen, fill it, and re-tap any cells the swipe-fill
    dropped until the win screen. Returns True on the win. This repair is
    non-destructive: it only adds taps for cells the solution says are filled
    but read back empty -- it never clears a cell, and never RESETs."""
    res = read_solvable()
    if not res:
        return False
    g, board = res
    n = g["n"]
    # Swipe-fill each run. Prime fill mode on the second cell so the first cell
    # isn't eaten by the mode-enter touch; no pencil button (it would fill the
    # cursor cell (0,0), wrong whenever board[0][0] is empty).
    auto.fill(board, g, fill_button=None, swipe_runs=True)
    for _ in range(3):
        if _won():
            return True
        actual = auto.read_board("/tmp/play.png", g, n=n)
        missing = [(i, j) for i in range(n) for j in range(n)
                   if board[i][j] == 1 and actual[i][j] != 1]
        for i, j in missing:
            auto.tap(*auto.cell_center(g, i, j))
    return _won()


def play_one():
    """Solve current level and advance to the next. Returns True on success."""
    if not in_puzzle(auto.screenshot("/tmp/play.png")):
        auto.start_next()                   # land in a puzzle if we aren't
    if not solve_current():
        return False
    return auto.start_next()


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    done = 0
    for k in range(count):
        print(f"[level {k + 1}/{count}]")
        if play_one():
            done += 1
            print(f"  done ({done} total)")
            time.sleep(0.5)
        else:
            print("  stopped: see /tmp/play.png")
            break
    print(f"completed {done}/{count} levels")
