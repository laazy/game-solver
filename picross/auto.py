"""Resolution-independent picross automation.

Merges the verified `picross_solver.Solver` with:
  - automatic grid detection (luminance projection, no hard-coded coords)
  - swipe-based fill (one swipe per run of filled cells)

Clue OCR is intentionally left to the caller: crop_clues() dumps the top/left
clue strips so they can be read (by a vision model or a classifier) and fed to
Solver.load_puzzle(col_info, row_info).

Env:
  ADB   path to adb binary (default "adb")
"""
import os
import json
import subprocess
from itertools import groupby

from PIL import Image, ImageDraw

from picross_solver import Solver

ADB = os.environ.get("ADB", "adb")

# Nonogram Galaxy menu layout for 1440x3200. These are the only hard-coded
# screen coords; the grid itself is detected. Re-measure per device/game.
NAV = {
    "win_continue": (718, 2417),      # checkmark on the victory screen
    "dialog_cancel": (1160, 1868),    # "取消" on the exit-confirm dialog
    "tile_xs": [143, 373, 603, 833, 1063, 1293],  # level-select tile columns
    "tile_ys": [2420, 2678],                       # level-select tile rows
    "tile_open_frac": 0.13,           # warm-fraction below -> unplayed tile
    "scroll_right": (1417, 2545),     # ">" to page the level list forward
}


# --------------------------------------------------------------------------- adb
def _sh(*args):
    return subprocess.run([ADB, *args], capture_output=True, text=True)


def screenshot(path="/tmp/picross.png"):
    _sh("shell", "screencap", "-p", "/sdcard/_ps.png")
    _sh("pull", "/sdcard/_ps.png", path)
    _sh("shell", "rm", "/sdcard/_ps.png")
    return path


def tap(x, y):
    _sh("shell", "input", "tap", str(int(x)), str(int(y)))


def swipe(x1, y1, x2, y2, ms=60):
    _sh("shell", "input", "swipe", str(int(x1)), str(int(y1)),
        str(int(x2)), str(int(y2)), str(ms))


# ------------------------------------------------------------------- grid detect
def _luma(p):
    r, g, b = p[:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _longest_run(mask, lo=0, max_gap=20):
    """Return (start, end) of the longest True run in mask, tolerating small
    gaps; ignore indices below lo. Returns (None, None) if no True."""
    best = (None, None)
    best_len = 0
    i = lo
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        gap = 0
        end = i
        while j < n and gap <= max_gap:
            if mask[j]:
                end = j
                gap = 0
            else:
                gap += 1
            j += 1
        if end - i > best_len:
            best_len = end - i
            best = (i, end)
        i = end + 1
    return best


def detect_grid(img_path, n=None, light_thr=100, fill_frac=0.55):
    """Find the playfield as the largest 'not dark' rectangle.

    Uses luminance projection: a grid row/col is bright across most of its
    span (cells are cream/tan, including the highlighted selection row/col),
    while clue digits are sparse and the header/margins are dark.

    Returns dict {x0,y0,x1,y1,cell,n}. If n (grid dimension) is None it is
    inferred from cell-spacing via the brightest square assumption; pass it
    explicitly when known for reliability.
    """
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    px = im.load()

    def bright(x, y):
        return _luma(px[x, y]) > light_thr

    # search the lower-central region to skip the top status bar
    y_lo, y_hi = int(H * 0.18), int(H * 0.72)
    x_lo = int(W * 0.05)

    # vertical projection -> x extent (longest contiguous bright band:
    # the playfield, not the clue digits or the yellow control buttons)
    span_y = range(y_lo, y_hi, 3)
    ny = len(span_y)
    xmask = [sum(bright(x, y) for y in span_y) > ny * fill_frac
             for x in range(W)]
    x0, x1 = _longest_run(xmask, x_lo)
    if x0 is None:
        raise RuntimeError("grid x-extent not found; lower fill_frac/light_thr")

    # horizontal projection within [x0,x1] -> y extent
    span_x = range(x0 + 5, x1 - 5, 3)
    nx = len(span_x)
    ymask = [sum(bright(x, y) for x in span_x) > nx * fill_frac
             for y in range(H)]
    y0, y1 = _longest_run(ymask, y_lo)
    if y0 is None:
        raise RuntimeError("grid y-extent not found")

    if n is None:
        n = _infer_n(im, x0, x1, y0, y1) or 10
    cell = ((x1 - x0) + (y1 - y0)) / (2 * n)
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "cell": cell, "n": n}


def _gridline_pitch(prof, min_gap=24):
    """Median spacing between gridlines (local brightness minima) in a 1-D
    profile. Robust to a few spurious minima and to X-marked cells."""
    import numpy as np
    thr = (np.percentile(prof, 20) + np.percentile(prof, 80)) / 2
    mins = []
    for i in range(1, len(prof) - 1):
        if prof[i] < thr and prof[i] <= prof[i - 1] and prof[i] <= prof[i + 1]:
            if not mins or i - mins[-1] >= min_gap:
                mins.append(i)
            elif prof[i] < prof[mins[-1]]:
                mins[-1] = i
    if len(mins) < 2:
        return None
    return float(np.median(np.diff(mins)))


def _infer_n(im, x0, x1, y0, y1):
    """Infer cell count as span / cell-size, where cell-size is the median
    gridline pitch. The median over the grid height dodges the highlighted
    cursor row/column, and the pitch survives X-marked (dimmer) cells."""
    import numpy as np
    arr = np.asarray(im.convert("RGB")).astype(np.float64)
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    colprof = np.median(luma[y0:y1, x0:x1], axis=0)
    rowprof = np.median(luma[y0:y1, x0:x1], axis=1)
    pitches = [p for p in (_gridline_pitch(colprof), _gridline_pitch(rowprof))
               if p]
    if not pitches:
        return None
    pitch = float(np.mean(pitches))
    return max(1, round(((x1 - x0) + (y1 - y0)) / 2 / pitch))


def cell_center(g, i, j):
    return (g["x0"] + g["cell"] * (j + 0.5),
            g["y0"] + g["cell"] * (i + 0.5))


def overlay(img_path, g, out="/tmp/picross_overlay.png", scale=2):
    im = Image.open(img_path).convert("RGB")
    d = ImageDraw.Draw(im)
    d.rectangle([g["x0"], g["y0"], g["x1"], g["y1"]],
                outline=(0, 255, 0), width=4)
    for i in range(g["n"]):
        for j in range(g["n"]):
            cx, cy = cell_center(g, i, j)
            d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7],
                      outline=(255, 0, 0), width=3)
    im.save(out)
    if scale > 1:
        im.resize((im.width // scale, im.height // scale)).save(
            out.replace(".png", "_s.png"))
    return out


def crop_clues(img_path, g, out_dir="/tmp"):
    """Dump the top (column) and left (row) clue strips for OCR/vision."""
    im = Image.open(img_path).convert("RGB")
    top = im.crop((g["x0"], 0, g["x1"], g["y0"]))
    left = im.crop((0, g["y0"], g["x0"], g["y1"]))
    tp = os.path.join(out_dir, "clues_top.png")
    lp = os.path.join(out_dir, "clues_left.png")
    top.save(tp)
    left.save(lp)
    return tp, lp


# ------------------------------------------------------------------------- fill
def _runs(board):
    """Yield (row, col_start, length) for every horizontal run of filled cells."""
    for i, row in enumerate(board):
        col = 0
        for value, group in groupby(row):
            length = len(list(group))
            if value == 1:
                yield i, col, length
            col += length


def fill(board, g, fill_button=None, first_tap=None, swipe_runs=True,
         ms_per_cell=45):
    """Fill the solved board on screen.

    fill_button: (x,y) of the pencil/fill-mode button to activate first.
    first_tap:   (x,y) tapped once before filling (some games need focus).
    swipe_runs:  one swipe per horizontal run of filled cells (faster);
                 single cells fall back to a tap.
    ms_per_cell: swipe duration scales with run length so a fast lift-off
                 doesn't drop the last cell of a run (observed failure mode).

    Tap-only fill (swipe_runs=False) primes fill mode with an extra tap on the
    first cell: the first tap on a fresh puzzle only enters fill mode and fills
    nothing, so without priming the first cell is dropped.
    """
    if fill_button:
        tap(*fill_button)
    if first_tap:
        tap(*first_tap)
    if not swipe_runs:
        cells = [(i, j) for i, col, length in _runs(board)
                 for j in range(col, col + length)]
        if cells:
            # The first tap on a fresh puzzle only *enters* fill mode (it moves
            # the cursor, fills nothing); every tap after that fills. So prime
            # fill mode with a tap on the first cell, fill the rest, then fill
            # that first cell LAST. Re-tapping it immediately would instead read
            # as a double-tap (the X-mark gesture); spacing the two taps apart
            # with the other cells avoids that and still fills it.
            first = cells[0]
            tap(*cell_center(g, *first))       # prime: enter fill mode only
            for i, j in cells[1:]:
                tap(*cell_center(g, i, j))
            tap(*cell_center(g, *first))       # fill the primed cell last
        return
    # Swipe-fill. The first touch on a fresh puzzle only *enters* fill mode
    # (fills nothing), so prime with a tap on the SECOND fill cell; the first
    # cell then fills normally when its run is drawn. A primed single cell is
    # filled last so the two taps on it don't read as a double-tap (X mark).
    runs = list(_runs(board))
    cells = [(i, j) for i, col, length in runs for j in range(col, col + length)]
    if not cells:
        return
    prime = cells[1] if len(cells) > 1 else cells[0]
    tap(*cell_center(g, *prime))               # enter fill mode (no fill)
    deferred = None
    for i, col, length in runs:
        if length == 1 and (i, col) == prime:
            deferred = (i, col)
            continue
        x0, y0 = cell_center(g, i, col)
        if length == 1:
            tap(x0, y0)
        else:
            x1, _ = cell_center(g, i, col + length - 1)
            swipe(x0, y0, x1, y0, ms=max(80, ms_per_cell * length))
    if deferred:
        tap(*cell_center(g, *deferred))


def read_board(img_path, g, n=None, dark_thr=90, x_thr=120):
    """Classify on-screen cell states from a screenshot.

    Returns a grid of 1 (filled/dark), -1 (X-marked), 0 (blank). Pass the
    grid geometry `g` explicitly — detect_grid() can't run on a filled board.
    """
    n = n or g.get("n", 10)
    im = Image.open(img_path).convert("RGB")
    px = im.load()
    r = max(6, int(g["cell"] * 0.3))   # patch stays inside the cell (no gridlines)
    step = max(2, r // 5)
    out = []
    for i in range(n):
        row = []
        for j in range(n):
            cx, cy = cell_center(g, i, j)
            cx, cy = int(cx), int(cy)
            vals = [_luma(px[cx + dx, cy + dy])
                    for dx in range(-r, r + 1, step) for dy in range(-r, r + 1, step)]
            avg = sum(vals) / len(vals)
            if avg < dark_thr:
                row.append(1)
            elif min(vals) < x_thr:
                row.append(-1)
            else:
                row.append(0)
        out.append(row)
    return out


def fill_verified(board, g, fill_button=None, first_tap=None, retries=3):
    """Fill, then screenshot-verify and re-tap any missing filled cells.

    Self-correcting against dropped swipe cells regardless of swipe speed.
    Returns the number of cells still missing after the last retry (0 == OK).
    """
    fill(board, g, fill_button=fill_button, first_tap=first_tap)
    n = len(board)
    for attempt in range(retries):
        shot = screenshot("/tmp/_verify.png")
        actual = read_board(shot, g, n=n)
        missing = [(i, j) for i in range(n) for j in range(n)
                   if board[i][j] == 1 and actual[i][j] != 1]
        if not missing:
            return 0
        for i, j in missing:
            tap(*cell_center(g, i, j))
    shot = screenshot("/tmp/_verify.png")
    actual = read_board(shot, g, n=n)
    return sum(1 for i in range(n) for j in range(n)
               if board[i][j] == 1 and actual[i][j] != 1)


def solve_board(col_info, row_info):
    import solver2
    board = solver2.solve(col_info, row_info)
    if board is None:
        raise AssertionError("no solution for the given clues")
    return board


# ------------------------------------------------------------------- navigation
def back():
    _sh("shell", "input", "keyevent", "KEYCODE_BACK")


def _warm_frac(px, cx, cy, r=85, step=8):
    """Fraction of warm/cream pixels over a tile. Completed-level tiles sit on
    a cream picture background; unplayed tiles are dark navy with white text.
    Luma alone fails here -- a mostly-black picture (e.g. a cat) is as dim as
    an unplayed tile, but its cream background still reads warm."""
    warm = total = 0
    for dx in range(-r, r + 1, step):
        for dy in range(-r, r + 1, step):
            R, G, B = px[cx + dx, cy + dy]
            total += 1
            if R > 150 and G > 140 and B > 120 and R >= B:
                warm += 1
    return warm / total


def find_next_tile(img_path):
    """On the level-select screen, return the (x,y) of the first unplayed
    tile (row-major). Completed tiles show a cream picture; unplayed tiles are
    dark with a dimension label. Returns None if none visible."""
    im = Image.open(img_path).convert("RGB")
    px = im.load()
    for y in NAV["tile_ys"]:
        for x in NAV["tile_xs"]:
            if _warm_frac(px, x, y) < NAV["tile_open_frac"]:
                return (x, y)
    return None


def _region_frac(img_path, y0, y1, x0, x1, test):
    import numpy as np
    a = np.asarray(Image.open(img_path).convert("RGB")).astype(np.int32)
    reg = a[y0:y1, x0:x1]
    R, G, B = reg[:, :, 0], reg[:, :, 1], reg[:, :, 2]
    return float(test(R, G, B).mean())


def in_puzzle(img_path):
    """True while a puzzle (with its yellow controls) is on screen."""
    return _region_frac(img_path, 2300, 2950, 0, 1440,
                        lambda R, G, B: (R > 180) & (G > 140) & (B < 110)) > 0.02


def exit_dialog(img_path):
    """True when the 'confirm exit?' dialog is up."""
    return _region_frac(img_path, 1480, 1780, 200, 1280,
                        lambda R, G, B: (R > 210) & (G > 200) & (B > 190)) > 0.3


def level_select(img_path):
    """True on the level-select screen (bottom tile grid present)."""
    return _region_frac(img_path, 2380, 2950, 0, 1440,
                        lambda R, G, B: (R > 120) | (G > 120)) > 0.05


def start_next(wait=1.5, max_steps=10):
    """From the victory screen, advance to the next puzzle, robust to the
    galaxy interstitial and the stray exit-confirm dialog.

    State machine: dialog -> cancel; level-select -> tap next tile; otherwise
    (victory/galaxy) -> continue then back. Returns True once in a puzzle."""
    import time
    for _ in range(max_steps):
        shot = screenshot("/tmp/_nav.png")
        if in_puzzle(shot):
            return True
        if exit_dialog(shot):
            tap(*NAV["dialog_cancel"])
            time.sleep(wait)
            continue
        if level_select(shot):
            nxt = find_next_tile(shot)
            if nxt:
                tap(*nxt)
                time.sleep(wait)
                continue
            # all visible levels done -> swipe the tile strip to page forward
            swipe(1100, 2545, 300, 2545, ms=400)
            time.sleep(wait)
            continue
        # victory or galaxy: back out toward level-select. Back works from
        # both and avoids the victory buttons, whose layout shifts (a replay
        # screen inserts a "retry" button where continue usually sits).
        back()
        time.sleep(wait)
    return in_puzzle(screenshot("/tmp/_nav.png"))


if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else screenshot()
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    g = detect_grid(img, n=n)
    print("grid:", json.dumps(g))
    overlay(img, g)
    crop_clues(img, g)
    print("overlay -> /tmp/picross_overlay_s.png ; clues -> /tmp/clues_{top,left}.png")
