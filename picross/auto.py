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
    "tile_open_frac": 0.18,           # light-fraction below -> unplayed tile
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


def detect_grid(img_path, n=None, light_thr=150, fill_frac=0.55):
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

    w, h = x1 - x0, y1 - y0
    if n is None:
        # Pixel span alone can't reveal the cell count; detect it from the
        # internal grid lines (dark/blue separators dip the brightness).
        n = _infer_n(px, x0, x1, y0, y1, light_thr) or 10
    cell = ((x1 - x0) + (y1 - y0)) / (2 * n)
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "cell": cell, "n": n}


def _infer_n(px, x0, x1, y0, y1, light_thr):
    """Count cells by finding the periodic gridline dips along the top edge."""
    yline = y0 + 3
    dark = []
    for x in range(x0, x1 + 1):
        if _luma(px[x, yline]) <= light_thr:
            dark.append(x)
    if not dark:
        return None
    # group contiguous dark pixels into gridline positions
    groups = []
    run = [dark[0]]
    for a, b in zip(dark, dark[1:]):
        if b - a <= 2:
            run.append(b)
        else:
            groups.append(sum(run) // len(run))
            run = [b]
    groups.append(sum(run) // len(run))
    # n cells -> n+1 separators (incl. borders); cells = gaps between groups
    return max(1, len(groups) - 1) if len(groups) >= 2 else None


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
    """
    if fill_button:
        tap(*fill_button)
    if first_tap:
        tap(*first_tap)
    for i, col, length in _runs(board):
        x0, y0 = cell_center(g, i, col)
        if length == 1 or not swipe_runs:
            for k in range(length):
                cx, cy = cell_center(g, i, col + k)
                tap(cx, cy)
        else:
            x1, _ = cell_center(g, i, col + length - 1)
            swipe(x0, y0, x1, y0, ms=max(80, ms_per_cell * length))


def read_board(img_path, g, n=None, dark_thr=90, x_thr=120):
    """Classify on-screen cell states from a screenshot.

    Returns a grid of 1 (filled/dark), -1 (X-marked), 0 (blank). Pass the
    grid geometry `g` explicitly — detect_grid() can't run on a filled board.
    """
    n = n or g.get("n", 10)
    im = Image.open(img_path).convert("RGB")
    px = im.load()
    out = []
    for i in range(n):
        row = []
        for j in range(n):
            cx, cy = cell_center(g, i, j)
            cx, cy = int(cx), int(cy)
            vals = [_luma(px[cx + dx, cy + dy])
                    for dx in range(-25, 26, 5) for dy in range(-25, 26, 5)]
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
    s = Solver()
    s.load_puzzle(col_info, row_info)
    s.solve()
    return s.board


# ------------------------------------------------------------------- navigation
def back():
    _sh("shell", "input", "keyevent", "KEYCODE_BACK")


def _light_frac(px, cx, cy, r=70, step=10, thr=150):
    vals = [_luma(px[cx + dx, cy + dy])
            for dx in range(-r, r + 1, step) for dy in range(-r, r + 1, step)]
    return sum(v > thr for v in vals) / len(vals)


def find_next_tile(img_path):
    """On the level-select screen, return the (x,y) of the first unplayed
    tile (row-major). Completed tiles show a bright picture; unplayed tiles
    are dark with a dimension label. Returns None if none visible."""
    im = Image.open(img_path).convert("RGB")
    px = im.load()
    for y in NAV["tile_ys"]:
        for x in NAV["tile_xs"]:
            if _light_frac(px, x, y) < NAV["tile_open_frac"]:
                return (x, y)
    return None


def start_next(wait=1.5):
    """From the victory screen, advance to the next puzzle.

    Flow: tap continue -> galaxy view -> back -> level select -> tap the
    next unplayed tile. Returns True once a puzzle screen is reached.
    Caller is responsible for the inter-step settle time via `wait`.
    """
    import time
    tap(*NAV["win_continue"])
    time.sleep(wait)
    back()                       # galaxy -> level select
    time.sleep(wait)
    shot = screenshot("/tmp/_levelsel.png")
    nxt = find_next_tile(shot)
    if not nxt:
        return False
    tap(*nxt)
    time.sleep(wait)
    return True


if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else screenshot()
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    g = detect_grid(img, n=n)
    print("grid:", json.dumps(g))
    overlay(img, g)
    crop_clues(img, g)
    print("overlay -> /tmp/picross_overlay_s.png ; clues -> /tmp/clues_{top,left}.png")
