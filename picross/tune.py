"""Fill-speed tuning loop: solve + fill only (no repair, no patient poll).

For each board: solve (keep the solution), fill with the params under test,
then immediately read back and report drops -- classified by their position in
the run (run-end drops mean the swipe lifts off too early; scattered drops mean
the whole swipe is too fast). Then tap the drops to finish the board and
advance, so the next iteration gets a fresh puzzle.

    python3 tune.py <ms_per_cell> <count>
"""
import sys
import time

import auto
import clue_ocr as C
import play


def _runs_of(board, n):
    out = []
    for i in range(n):
        j = 0
        while j < n:
            if board[i][j] == 1:
                k = j
                while k < n and board[i][k] == 1:
                    k += 1
                out.append((i, j, k - j))
                j = k
            else:
                j += 1
    return out


def classify(drops, board, n):
    runs = {(i, c): L for i, c, L in _runs_of(board, n)}
    cell_run = {}
    for (i, c), L in runs.items():
        for d in range(L):
            cell_run[(i, c + d)] = (c, L, d)        # start, length, offset
    tags = {"end": 0, "start": 0, "mid": 0, "single": 0}
    for i, j in drops:
        c, L, d = cell_run[(i, j)]
        if L == 1:
            tags["single"] += 1
        elif d == L - 1:
            tags["end"] += 1
        elif d == 0:
            tags["start"] += 1
        else:
            tags["mid"] += 1
    return tags


def fresh_board(tries=8):
    """Return (img, g, n, board) for a solvable on-screen puzzle, advancing
    past unreadable/in-progress screens."""
    for _ in range(tries):
        img = auto.screenshot("/tmp/t.png")
        if play.in_puzzle(img):
            try:
                g = auto.detect_grid(img, n=None)
                col, row, board = C.solvable_clues(img, g)
                return img, g, g["n"], board
            except (AssertionError, RuntimeError):
                pass                                  # unreadable -> advance
        auto.start_next()
        time.sleep(1.3)
    raise RuntimeError("no fresh solvable board found")


def one(ms_pc):
    img, g, n, board = fresh_board()
    t0 = time.time()
    auto.fill(board, g, fill_button=None, swipe_runs=True, ms_per_cell=ms_pc)
    ft = time.time() - t0
    time.sleep(0.7)
    img2 = auto.screenshot("/tmp/t2.png")
    if not play.in_puzzle(img2):
        print(f"n={n:2d} ms={ms_pc:2d} fill={ft:4.1f}s  ONE-PASS OK")
        auto.start_next()
        return
    a = auto.read_board("/tmp/t2.png", g, n=n)
    drops = [(i, j) for i in range(n) for j in range(n)
             if board[i][j] == 1 and a[i][j] != 1]
    tags = classify(drops, board, n)
    print(f"n={n:2d} ms={ms_pc:2d} fill={ft:4.1f}s  DROPS={len(drops):2d} "
          f"{tags}  {drops[:12]}")
    for i, j in drops:                                # finish so we can advance
        auto.tap(*auto.cell_center(g, i, j))
    time.sleep(0.5)
    auto.start_next()


if __name__ == "__main__":
    ms = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    for _ in range(count):
        try:
            one(ms)
        except AssertionError as e:
            print("OCR fail:", e)
            auto.start_next()
        time.sleep(1.0)
