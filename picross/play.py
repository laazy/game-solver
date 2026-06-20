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


def solve_current():
    """Solve the puzzle currently on screen. Returns True once the win screen
    is reached, False if OCR/solve failed (screen left untouched-ish)."""
    img = auto.screenshot("/tmp/play.png")
    g = auto.detect_grid(img, n=None)
    try:
        col, row, board = clue_ocr.solvable_clues(img, g)
    except AssertionError as e:
        print(f"  ! {e}; skipping fill")
        return False

    # Enter fill mode by tapping the first solution cell directly. Tapping the
    # pencil *button* instead would fill the cursor cell (0,0) — wrong whenever
    # board[0][0] is empty. tap-only (no swipes: those fling past a run).
    auto.fill(board, g, fill_button=None, swipe_runs=False)

    n = g["n"]

    def won():
        for _ in range(4):
            time.sleep(1.0)
            if not in_puzzle(auto.screenshot("/tmp/play.png")):
                return True
        return False

    # trust the tap-only fill first; only repair genuinely missed cells.
    # We never remove "extra" cells: filled cells read back reliably (dark),
    # so missing has no false positives, whereas an empty cell under the
    # cursor can read as filled -> removing it would wrongly fill (0,0).
    for _ in range(3):
        if won():
            return True
        actual = auto.read_board("/tmp/play.png", g, n=n)
        missing = [(i, j) for i in range(n) for j in range(n)
                   if board[i][j] == 1 and actual[i][j] != 1]
        for i, j in missing:
            auto.tap(*auto.cell_center(g, i, j))
    return won()                            # final patient poll for the win


def play_one():
    """Solve current level and advance to the next. Returns True on success."""
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
            time.sleep(1.5)
        else:
            print("  stopped: see /tmp/play.png")
            break
    print(f"completed {done}/{count} levels")
