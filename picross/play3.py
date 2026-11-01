"""play3.py -- solve the single picross level on screen, via adb, with solver3c.

Same device pipeline as play.py (auto.py adb/grid/OCR/fill + clue_ocr), but the
board is produced by our solver, solver3c (ordered DP propagation + backtracking),
and it only ever touches the puzzle currently on screen -- no level navigation.

    screenshot -> detect grid (auto size) -> OCR clues -> solve (solver3c,
    falling back to clue_ocr's OCR-repair / line-recovery when raw OCR clues
    are inconsistent) -> swipe-fill -> verify & re-tap dropped cells until win.

Usage:
    ADB=/path/to/adb python3 play3.py            # solve the level on screen
    python3 play3.py --dry puzzles/1-46.txt      # offline: solve & print fill plan
"""
import os
import sys
import time

import auto
import clue_ocr
import recover_board
import solver3c

# ---- debug instrumentation (set PLAY3_DEBUG=1) ----
DEBUG = os.environ.get("PLAY3_DEBUG") == "1"
FIRST_PASS_ONLY = os.environ.get("PLAY3_FIRSTPASS") == "1"
_T0 = time.perf_counter()


def _log(msg):
    if DEBUG:
        print(f"[{time.perf_counter() - _T0:7.2f}s] {msg}", flush=True)


def _instrument_adb():
    """Wrap auto.tap/swipe/screenshot to log every call + its duration, so a
    hanging adb call shows as a 'start' line with no matching 'done'."""
    if not DEBUG:
        return
    _tap, _swipe, _shot = auto.tap, auto.swipe, auto.screenshot

    def tap(x, y):
        _log(f"  tap({int(x)},{int(y)}) ...")
        t = time.perf_counter()
        _tap(x, y)
        _log(f"  tap done in {time.perf_counter() - t:.2f}s")

    def swipe(x1, y1, x2, y2, ms=60):
        _log(f"  swipe({int(x1)},{int(y1)}->{int(x2)},{int(y2)}) ...")
        t = time.perf_counter()
        _swipe(x1, y1, x2, y2, ms)
        _log(f"  swipe done in {time.perf_counter() - t:.2f}s")

    def screenshot(path="/tmp/picross.png"):
        _log(f"  screenshot({path}) ...")
        t = time.perf_counter()
        r = _shot(path)
        _log(f"  screenshot done in {time.perf_counter() - t:.2f}s")
        return r

    auto.tap, auto.swipe, auto.screenshot = tap, swipe, screenshot


# --------------------------------------------------------------------- solving
def solve_clues(img, g):
    """Return (col_info, row_info, board) for the puzzle on screen, or None.

    The clues returned are the ones the board was actually solved from (possibly
    OCR-repaired), so the pass-by-pass replay deduces from consistent clues.

    Try our solver on the raw OCR clues first; if those clues don't solve
    (almost always an OCR digit error, not an unsolvable puzzle), fall back to
    clue_ocr's confidence-aware digit repair, then to line recovery -- exactly
    the ladder play.py uses, so we inherit its OCR robustness."""
    cols, rows = clue_ocr.read_clues(img, g)
    board = solver3c.solve(cols, rows)
    if board is not None:
        return cols, rows, board

    print("  ! raw OCR clues don't solve; trying digit repair")
    try:
        cols, rows, board = clue_ocr.solvable_clues(img, g)
        return cols, rows, board
    except AssertionError as e:
        print(f"  ! {e}; trying line-recovery")
        out = recover_board.recover(cols, rows, max_free=2)
        if out is None:
            print("  ! recovery failed")
            return None
        cols, rows, board = out
        print("  recovered board via free-line search")
        return cols, rows, board


def propagation_passes(col_info, row_info, solution):
    """Replay the row/column line-propagation the way the algorithm iterates and
    yield, per pass, the cells it newly resolves to FILLED (value 1).

    Pure alternating passes (all rows, then all columns, repeat) using the DP
    forced-cell resolver -- the same deductions the solver makes, surfaced one
    pass at a time so the fill can mirror them. After propagation stalls, a final
    'guess' batch covers any filled cells only the full solution (backtracking)
    pins down, so guess-requiring puzzles still complete.
    """
    from solver2 import _line_forced
    n, m = len(row_info), len(col_info)
    board = [[0] * m for _ in range(n)]
    seen = set()                                   # filled cells already yielded

    def newly_filled():
        out = []
        for i in range(n):
            for j in range(m):
                if board[i][j] == 1 and (i, j) not in seen:
                    seen.add((i, j))
                    out.append((i, j))
        return out

    kinds = ("row", "col")
    p, stall, last = 0, 0, 0
    while True:
        kind = kinds[p % 2]
        if kind == "row":
            for i in range(n):
                res = _line_forced(row_info[i], board[i])
                if res:
                    board[i] = res
        else:
            for j in range(m):
                res = _line_forced(col_info[j], [board[i][j] for i in range(n)])
                if res:
                    for i in range(n):
                        board[i][j] = res[i]
        yield kind, newly_filled()
        known = sum(1 for r in board for c in r if c != 0)
        if known == n * m:
            return
        stall = stall + 1 if known == last else 0
        last = known
        p += 1
        if stall >= 2:                             # a row+col pass with no change
            break
    rest = [(i, j) for i in range(n) for j in range(m)
            if solution[i][j] == 1 and (i, j) not in seen]
    if rest:
        yield "guess", rest


def _runs_from_cells(cells, vertical=False):
    """Group cells into maximal straight runs, each (row0, col0, length).

    vertical=False: horizontal runs (contiguous columns in a row) -- for row
    passes. vertical=True: vertical runs (contiguous rows in a column), ordered
    column by column then top-to-bottom -- so a column pass fills down each
    column instead of scattering single taps across rows."""
    runs = []
    if vertical:
        for j, i in sorted((c, r) for r, c in cells):      # by column, then row
            if runs and runs[-1][1] == j and runs[-1][0] + runs[-1][2] == i:
                r0, c0, L = runs[-1]
                runs[-1] = (r0, c0, L + 1)
            else:
                runs.append((i, j, 1))
    else:
        for i, j in sorted(cells):                         # by row, then column
            if runs and runs[-1][0] == i and runs[-1][1] + runs[-1][2] == j:
                r0, c0, L = runs[-1]
                runs[-1] = (r0, c0, L + 1)
            else:
                runs.append((i, j, 1))
    return runs


def _draw_run(g, i, c, length, vertical=False, ms_per_cell=45):
    """Fill one straight run with a single swipe (or a tap for length 1)."""
    x0, y0 = auto.cell_center(g, i, c)
    if length == 1:
        auto.tap(x0, y0)
    elif vertical:
        _, y1 = auto.cell_center(g, i + length - 1, c)
        auto.swipe(x0, y0, x0, y1, ms=max(80, ms_per_cell * length))
    else:
        x1, _ = auto.cell_center(g, i, c + length - 1)
        auto.swipe(x0, y0, x1, y0, ms=max(80, ms_per_cell * length))


def fill_incremental(passes, g, pause=0.4, confirm=False):
    """Fill the board pass by pass using swipe-per-run -- the reliable gesture
    the repo settled on -- so the row/column propagation is visible without the
    input-queue corruption that tap-per-cell causes over (network) adb.

    The first gesture on a fresh puzzle only enters fill mode (fills nothing),
    so prime on the first run's first cell and fill that one cell once enough
    other gestures have spaced the two taps apart (else it reads as a double-tap
    X-mark).

    confirm=True: wait for the user to press Enter between passes, so they can
    verify each pass on the phone before the next one runs."""
    entered = False
    deferred = None                                    # primed cell, fill later
    for pass_no, (kind, cells) in enumerate(passes):
        vertical = (kind == "col")                     # column passes fill downward
        runs = _runs_from_cells(cells, vertical=vertical)
        _log(f"{kind} pass #{pass_no}: {len(cells)} cells, {len(runs)} runs")
        for (i, c, length) in runs:
            if not entered:                            # very first gesture
                auto.tap(*auto.cell_center(g, i, c))   # enter fill mode (no fill)
                entered = True
                deferred = (i, c)
                if length > 1:                         # fill the rest of this run
                    if vertical:
                        _draw_run(g, i + 1, c, length - 1, vertical=True)
                    else:
                        _draw_run(g, i, c + 1, length - 1, vertical=False)
            else:
                _draw_run(g, i, c, length, vertical=vertical)
            time.sleep(0.05)
        if deferred and len(runs) > 1:                 # spaced enough -> safe now
            auto.tap(*auto.cell_center(g, *deferred))
            deferred = None
        print(f"  {kind} pass: +{len(cells)} filled ({len(runs)} runs)", flush=True)
        if FIRST_PASS_ONLY:
            print("  PLAY3_FIRSTPASS: exiting after first pass", flush=True)
            if deferred:
                auto.tap(*auto.cell_center(g, *deferred))
            sys.exit(0)
        if confirm:
            try:
                input("  >> pass filled on phone. Enter = next pass, Ctrl-C = stop ")
            except (EOFError, KeyboardInterrupt):
                print("\n  stopped by user", flush=True)
                return
        else:
            time.sleep(pause)
    if deferred:
        auto.tap(*auto.cell_center(g, *deferred))      # fill the primed cell last


def solve_current(step=False):
    """Solve and fill the level on screen. True once the win screen shows.

    step=True: pause for confirmation between propagation passes."""
    _instrument_adb()
    _log("screenshot ...")
    img = auto.screenshot("/tmp/play3.png")
    _log("detect_grid ...")
    try:
        g = auto.detect_grid(img, n=None)
    except RuntimeError:
        print("  ! no grid on screen; open a puzzle first")
        return False
    _log(f"grid n={g['n']}")

    _log("solving ...")
    solved = solve_clues(img, g)
    if solved is None:
        print("  ! could not solve; skipping fill")
        return False
    cols, rows, board = solved

    n = g["n"]
    print(f"  solved {n}x{n}; filling pass by pass", flush=True)
    _log("fill_incremental start")
    fill_incremental(propagation_passes(cols, rows, board), g, confirm=step)
    _log("fill_incremental done")

    def won():
        for _ in range(4):
            time.sleep(1.0)
            if not auto.in_puzzle(auto.screenshot("/tmp/play3.png")):
                return True
        return False

    # Trust the fill, then repair only genuinely missed (still-blank) filled
    # cells. We never *remove* extras: a filled cell reads back reliably, but an
    # empty cell under the cursor can read as filled, so removal has false
    # positives while "missing" does not. (Same rationale as play.py.)
    for _ in range(3):
        if won():
            return True
        actual = auto.read_board("/tmp/play3.png", g, n=n)
        missing = [(i, j) for i in range(n) for j in range(n)
                   if board[i][j] == 1 and actual[i][j] != 1]
        print(f"  re-tapping {len(missing)} missed cells")
        for i, j in missing:
            auto.tap(*auto.cell_center(g, i, j))
    return won()


# ---------------------------------------------------------------------- offline
def _load_puzzle(path):
    if path.endswith(".json"):
        import json
        d = json.load(open(path))
        return d["col"], d["row"]
    with open(path) as f:
        f.readline()                                   # "rows cols" header
        col = [[int(x) for x in g.split()] for g in f.readline().strip().split(",")]
        row = [[int(x) for x in g.split()] for g in f.readline().strip().split(",")]
    return col, row


def dry_run(path):
    """Solve a puzzle file and print the board + the horizontal fill runs that
    fill() would swipe -- verifies the solver/fill plan with no device."""
    col, row = _load_puzzle(path)
    t = time.perf_counter()
    board = solver3c.solve(col, row)
    dt = (time.perf_counter() - t) * 1000
    if board is None:
        print(f"{path}: UNSATISFIABLE clues")
        return 1
    n, m = len(board), len(board[0])
    print(f"{path}  {n}x{m}  solved in {dt:.0f}ms\n")
    for r in board:
        print("".join("#" if c == 1 else "." for c in r))
    runs = list(auto._runs(board))
    print(f"\nfill plan: {len(runs)} swipe runs "
          f"({sum(1 for _, _, L in runs if L > 1)} swipes, "
          f"{sum(1 for _, _, L in runs if L == 1)} taps)")
    return 0


# --------------------------------------------------------------------- entry
if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--dry":
        sys.exit(dry_run(args[1]))

    step = "--step" in args                  # pause for Enter between passes
    ok = solve_current(step=step)
    print("done" if ok else "stopped: see /tmp/play3.png")
    sys.exit(0 if ok else 1)
