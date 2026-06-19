"""Code-only OCR for picross clue digits.

Fixed printed font on a clean background -> template matching is reliable.
Digits are segmented by connected components, normalized to a 28x28 bitmap,
and classified by signed correlation against per-digit mean templates.

Glyphs are grouped into clue numbers by position:
  - column clues stack vertically; glyphs sharing a y-level form one number
  - row clues run horizontally; glyphs with a small x-gap form one number

The model lives in clue_model.npz (keys + mean templates). Rebuild/extend it
with add_samples()/save_model() as new digits appear.
"""
import os
import numpy as np
from PIL import Image
from scipy import ndimage

MODEL_PATH = os.path.join(os.path.dirname(__file__), "clue_model.npz")
S = 28
GAP_SAME_NUMBER = 16          # x-gap (px) below which row-clue glyphs join
THR = 140                     # luma threshold for white digits


# ------------------------------------------------------------------ primitives
def luma_of(img_path):
    a = np.asarray(Image.open(img_path).convert("RGB")).astype(np.int32)
    return 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]


def _normalize(bmp):
    im = Image.fromarray((bmp * 255).astype("uint8"))
    iw, ih = im.size
    sc = min((S - 4) / iw, (S - 4) / ih)
    nw, nh = max(1, int(iw * sc)), max(1, int(ih * sc))
    im = im.resize((nw, nh), Image.NEAREST)
    canvas = Image.new("L", (S, S), 0)
    canvas.paste(im, ((S - nw) // 2, (S - nh) // 2))
    return np.asarray(canvas) > 127


def extract_glyphs(luma, x0, x1, y0, y1, thr=THR):
    """Connected-component glyph boxes within a region, with normalized bmp."""
    binm = luma[y0:y1, x0:x1] > thr
    lbl, _ = ndimage.label(binm)
    out = []
    for sl in ndimage.find_objects(lbl):
        ys, xs = sl
        h, w = ys.stop - ys.start, xs.stop - xs.start
        if h < 12 or h > 60 or w < 4 or w > 60 or int(binm[sl].sum()) < 20:
            continue
        out.append({
            "x0": x0 + xs.start, "x1": x0 + xs.stop,
            "cx": x0 + xs.start + w / 2, "cy": y0 + ys.start + h / 2,
            "norm": _normalize(binm[sl]),
        })
    return out


# ----------------------------------------------------------------------- model
def _signed(b):
    return b.astype(np.float32) * 2 - 1


def build_model(samples):
    """samples: list of (norm_bitmap, label). Returns (keys, templates)."""
    by_digit = {}
    for bmp, lab in samples:
        by_digit.setdefault(int(lab), []).append(_signed(bmp))
    keys = sorted(by_digit)
    templates = np.stack([np.mean(by_digit[k], axis=0) for k in keys])
    return np.array(keys), templates


def save_model(keys, templates, path=MODEL_PATH):
    np.savez(path, keys=keys, templates=templates)


def load_model(path=MODEL_PATH):
    z = np.load(path)
    return z["keys"], z["templates"]


class Classifier:
    def __init__(self, path=MODEL_PATH):
        self.keys, self.templates = load_model(path)

    def classify(self, norm_bmp, forbid=()):
        s = _signed(norm_bmp)
        scores = (self.templates * s).reshape(len(self.keys), -1).sum(axis=1)
        order = scores.argsort()[::-1]
        for idx in order:
            d = int(self.keys[int(idx)])
            if d not in forbid:
                return d
        return int(self.keys[int(order[0])])


# -------------------------------------------------------------------- grouping
def _group_by_slot(glyphs, axis, ref, pitch):
    """Group stacked column clues by slot along `axis` ('cy').

    Clue numbers sit on a regular pitch (~cell/2) stepping away from the grid;
    each number occupies one slot (a two-digit number's digits share a y-level
    -> same slot). Returns numbers top->bottom (slot farthest from grid first)."""
    by_slot = {}
    for g in glyphs:
        slot = round((ref - g[axis]) / pitch)
        by_slot.setdefault(slot, []).append(g)
    return [by_slot[s] for s in sorted(by_slot, reverse=True)]


def _group_row(glyphs, x_ref, pitch, gap_merge=8):
    """Group horizontal row clues left->right.

    Two glyphs belong to the same number when they share a slot OR nearly
    touch (gap < gap_merge). Slots alone split a two-digit number that is
    wider than one slot (tight 20x20 grids); the touch rule rejoins it, while
    the slot rule keeps distinct-but-close single-digit clues apart."""
    glyphs = sorted(glyphs, key=lambda d: d["cx"])
    if not glyphs:
        return []
    numbers = [[glyphs[0]]]
    for prev, g in zip(glyphs, glyphs[1:]):
        same_slot = (round((x_ref - prev["cx"]) / pitch)
                     == round((x_ref - g["cx"]) / pitch))
        touching = g["x0"] - prev["x1"] < gap_merge
        if same_slot or touching:
            numbers[-1].append(g)
        else:
            numbers.append([g])
    return numbers


def _drop_header(glyphs, max_gap=70):
    """Keep only the clue block nearest the grid, dropping the title/HINT row.

    Walking upward from the grid, stop at the first large vertical gap."""
    if not glyphs:
        return glyphs
    ordered = sorted(glyphs, key=lambda d: d["cy"], reverse=True)
    kept = [ordered[0]]
    for g in ordered[1:]:
        if kept[-1]["cy"] - g["cy"] > max_gap:
            break
        kept.append(g)
    return kept


def _digits_to_number(group, clf, forbid_zero=False):
    group = sorted(group, key=lambda d: d["cx"])
    val = 0
    for k, g in enumerate(group):
        # a clue digit is never a leading 0; a standalone single 0 among other
        # clues is always a misread (0/9 look alike) -> take best non-zero
        forbid = (0,) if (forbid_zero or k == 0) and len(group) == 1 else ()
        val = val * 10 + clf.classify(g["norm"], forbid=forbid)
    return val


# ------------------------------------------------------------------ public API
def read_clues(img_path, g, clf=None):
    """Return (col_info, row_info) read from a fresh-puzzle screenshot.

    g: grid dict from auto.detect_grid (needs x0,y0,x1,y1,cell,n).
    """
    clf = clf or Classifier()
    lu = luma_of(img_path)
    n = g["n"]
    cell = g["cell"]

    # column clues: region above the grid, within its x-span
    top = _drop_header(extract_glyphs(lu, int(g["x0"]), int(g["x1"]),
                                      0, int(g["y0"])))
    cols = [[] for _ in range(n)]
    for gl in top:
        j = int(round((gl["cx"] - g["x0"]) / cell - 0.5))
        if 0 <= j < n:
            cols[j].append(gl)
    pitch = cell / 2

    def read_line(groups):
        multi = len(groups) > 1
        nums = [_digits_to_number(grp, clf, forbid_zero=multi)
                for grp in groups]
        # a single clue can never exceed the line length n; a too-big value is
        # an over-merge of adjacent clues -> split it back into its digits
        out = []
        for v in nums:
            out.extend(int(d) for d in str(v)) if v > n else out.append(v)
        return out or [0]

    col_info = [read_line(_group_by_slot(c, "cy", g["y0"], pitch)) for c in cols]

    # row clues: region left of the grid, within its y-span
    left = extract_glyphs(lu, 0, int(g["x0"]), int(g["y0"]), int(g["y1"]))
    rows = [[] for _ in range(n)]
    for gl in left:
        i = int(round((gl["cy"] - g["y0"]) / cell - 0.5))
        if 0 <= i < n:
            rows[i].append(gl)
    row_info = [read_line(_group_row(r, g["x0"], pitch)) for r in rows]

    return col_info, row_info


def clues_consistent(col_info, row_info):
    """Sanity check: total filled cells implied by rows == by columns."""
    return sum(sum(c) for c in col_info) == sum(sum(r) for r in row_info)


def _merge_variants(info, n):
    """Yield copies of `info` (list of lines) with one adjacent clue pair in
    one line merged into a two-digit number (<= n). Recovers a two-digit clue
    that OCR split into separate digits."""
    for li, line in enumerate(info):
        for k in range(len(line) - 1):
            merged = int(f"{line[k]}{line[k + 1]}")
            if merged <= n:
                new = list(line[:k]) + [merged] + list(line[k + 2:])
                yield [new if i == li else l for i, l in enumerate(info)]


def solvable_clues(img_path, g, clf=None):
    """Read clues and return (col_info, row_info, board), repairing a single
    split/merge OCR error using the solver as the oracle. Raises if no
    consistent, solvable reading is found."""
    import solver2
    col, row = read_clues(img_path, g, clf)
    n = g["n"]
    if clues_consistent(col, row):
        b = solver2.solve(col, row)
        if b:
            return col, row, b
    # one mis-split two-digit clue: try merging a pair in some row or column
    for nrow in _merge_variants(row, n):
        if clues_consistent(col, nrow):
            b = solver2.solve(col, nrow)
            if b:
                return col, nrow, b
    for ncol in _merge_variants(col, n):
        if clues_consistent(ncol, row):
            b = solver2.solve(ncol, row)
            if b:
                return ncol, row, b
    raise AssertionError(
        f"unsolvable OCR (col_sum={sum(map(sum, col))} "
        f"row_sum={sum(map(sum, row))})")
