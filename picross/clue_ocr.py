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


def _split_wide(box, x0, y0, unit):
    """Split a too-wide glyph box (touching digits, common on 25x25) at the
    sparsest interior column(s) into `unit`-wide pieces. box is (bmp, bx, by)."""
    bmp, bx, by = box
    w = bmp.shape[1]
    k = round(w / unit)
    if k < 2:
        return [box]
    colsum = bmp.sum(axis=0)
    cuts, lo = [], 0
    for part in range(1, k):
        target = round(part * w / k)
        win = range(max(lo + 4, target - 6), min(w - 4, target + 7))
        c = min(win, key=lambda c: colsum[c], default=target)
        cuts.append(c)
        lo = c
    pieces, prev = [], 0
    for c in cuts + [w]:
        sub = bmp[:, prev:c]
        if sub.any():
            pieces.append((sub, bx + prev, by))
        prev = c
    return pieces


def extract_glyphs(luma, x0, x1, y0, y1, thr=THR):
    """Connected-component glyph boxes within a region, with normalized bmp.
    Over-wide components (adjacent digits that touch on small grids) are split
    into single digits at their sparsest columns."""
    binm = luma[y0:y1, x0:x1] > thr
    lbl, _ = ndimage.label(binm)
    boxes = []
    for sl in ndimage.find_objects(lbl):
        ys, xs = sl
        h, w = ys.stop - ys.start, xs.stop - xs.start
        if h < 12 or h > 60 or w < 4 or w > 60 or int(binm[sl].sum()) < 20:
            continue
        boxes.append((binm[sl], xs.start, ys.start))
    if not boxes:
        return []
    unit = sorted(b[0].shape[1] for b in boxes)[len(boxes) // 2]   # median width
    out = []
    for box in boxes:
        # a single digit is always taller than wide (w/h <= ~0.8); a box that
        # is near-square or wider is two touching digits -> split it
        wide = box[0].shape[1] > 0.95 * box[0].shape[0]
        for bmp, bx, by in (_split_wide(box, x0, y0, unit) if wide else [box]):
            out.append({
                "x0": x0 + bx, "x1": x0 + bx + bmp.shape[1],
                "cx": x0 + bx + bmp.shape[1] / 2, "cy": y0 + by + bmp.shape[0] / 2,
                "norm": _normalize(bmp),
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

    def ranked(self, norm_bmp):
        """Digits ranked best-first, as [(digit, score), ...]."""
        s = _signed(norm_bmp)
        scores = (self.templates * s).reshape(len(self.keys), -1).sum(axis=1)
        return [(int(self.keys[int(i)]), float(scores[int(i)]))
                for i in scores.argsort()[::-1]]

    def classify(self, norm_bmp, forbid=()):
        for d, _ in self.ranked(norm_bmp):
            if d not in forbid:
                return d
        return self.ranked(norm_bmp)[0][0]


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


def _group_row(glyphs, x_ref, pitch, clf, n, gap_merge=8):
    """Group horizontal row clues left->right.

    Two glyphs join the same number when they share a slot OR nearly touch
    (gap < gap_merge) AND the resulting value stays <= n (a clue can't exceed
    the line length). The <=n guard is essential on tight 25x25 grids where
    even distinct clues sit a few px apart: '5','1','0' must parse as '5','10'
    (51>25 rejected) rather than merge into an impossible '510'.

    Returns (numbers, split_gaps) where split_gaps[i] is the px gap at the
    boundary after number i -- small gaps are likely mis-splits worth merging."""
    glyphs = sorted(glyphs, key=lambda d: d["cx"])
    if not glyphs:
        return [], []
    for g in glyphs:
        g["_d"] = clf.classify(g["norm"])
    numbers = [[glyphs[0]]]
    split_gaps = []
    for prev, g in zip(glyphs, glyphs[1:]):
        gap = g["x0"] - prev["x1"]
        same_slot = (round((x_ref - prev["cx"]) / pitch)
                     == round((x_ref - g["cx"]) / pitch))
        cur = int("".join(str(x["_d"]) for x in numbers[-1]))
        merged = int(f"{cur}{g['_d']}")
        if (same_slot or gap < gap_merge) and merged <= n:
            numbers[-1].append(g)
        else:
            numbers.append([g])
            split_gaps.append(gap)
    return numbers, split_gaps


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
    """Return (col_info, row_info). See read_clues_full for merge candidates."""
    col_info, row_info, _ = read_clues_full(img_path, g, clf)
    return col_info, row_info


SWAP_MARGIN = 40          # score gap below which a glyph's reading is uncertain


def _line_edits(side, idx, groups, gaps, n, clf):
    """Read one line's clue values and the candidate repair edits for it.

    Each edit is (cost, side, idx, new_line): a smaller cost is a likelier
    OCR error. Edits cover mis-splits (re-merge two clues -- cost is the px
    gap) and uncertain digits (re-classify to the 2nd-best -- cost is the
    score margin). Returns (values, edits)."""
    multi = len(groups) > 1
    ranked, igaps, nums = [], [], []
    for grp in groups:
        grp = sorted(grp, key=lambda d: d["cx"])
        rs = []
        for k, gl in enumerate(grp):
            forbid = (0,) if (multi or k == 0) and len(grp) == 1 else ()
            r = [t for t in clf.ranked(gl["norm"]) if t[0] not in forbid]
            rs.append(r or clf.ranked(gl["norm"]))
        ranked.append(rs)
        igaps.append([grp[k + 1]["x0"] - grp[k]["x1"]
                      for k in range(len(grp) - 1)])   # gaps within the number
        nums.append(int("".join(str(r[0][0]) for r in rs)))

    if any(v > n for v in nums):            # >n split shifts positions
        out = []
        for v in nums:
            out.extend(int(d) for d in str(v)) if v > n else out.append(v)
        return out or [0], []

    values = nums or [0]
    edits = []
    for pos, gap in enumerate(gaps):        # re-merge a mis-split
        merged = int(f"{values[pos]}{values[pos + 1]}")
        if merged <= n:
            edits.append((gap, side, idx,
                          values[:pos] + [merged] + values[pos + 2:]))
    for ci, rs in enumerate(ranked):        # re-split an over-merge (rarer ->
        digs = [str(t[0][0]) for t in rs]   # cost penalty so it ranks last)
        for p, ig in enumerate(igaps[ci], start=1):
            left, right = int("".join(digs[:p])), int("".join(digs[p:]))
            edits.append((ig + 30, side, idx,
                          values[:ci] + [left, right] + values[ci + 1:]))
    for ci, rs in enumerate(ranked):        # re-classify an uncertain digit
        for p, r in enumerate(rs):
            if len(r) < 2 or r[0][1] - r[1][1] > SWAP_MARGIN:
                continue
            digs = [str(t[0][0]) for t in rs]
            digs[p] = str(r[1][0])
            alt = int("".join(digs))
            if 1 <= alt <= n:
                edits.append((r[0][1] - r[1][1], side, idx,
                              values[:ci] + [alt] + values[ci + 1:]))
    return values, edits


def read_clues_full(img_path, g, clf=None):
    """Return (col_info, row_info, edits): clue lists plus candidate repair
    edits (mis-split re-merges and uncertain-digit swaps), sorted likeliest
    first, for the solver-oracle repair."""
    clf = clf or Classifier()
    lu = luma_of(img_path)
    n, cell = g["n"], g["cell"]
    pitch = cell / 2

    top = _drop_header(extract_glyphs(lu, int(g["x0"]), int(g["x1"]),
                                      0, int(g["y0"])))
    cols = [[] for _ in range(n)]
    for gl in top:
        j = int(round((gl["cx"] - g["x0"]) / cell - 0.5))
        if 0 <= j < n:
            cols[j].append(gl)

    left = extract_glyphs(lu, 0, int(g["x0"]), int(g["y0"]), int(g["y1"]))
    rows = [[] for _ in range(n)]
    for gl in left:
        i = int(round((gl["cy"] - g["y0"]) / cell - 0.5))
        if 0 <= i < n:
            rows[i].append(gl)

    col_info, row_info, edits = [], [], []
    for j, c in enumerate(cols):
        vals, e = _line_edits("col", j, _group_by_slot(c, "cy", g["y0"], pitch),
                              [], n, clf)
        col_info.append(vals)
        edits += e
    for i, r in enumerate(rows):
        groups, gaps = _group_row(r, g["x0"], pitch, clf, n)
        vals, e = _line_edits("row", i, groups, gaps, n, clf)
        row_info.append(vals)
        edits += e

    edits.sort(key=lambda e: e[0])          # smallest cost first = likeliest
    return col_info, row_info, edits


def clues_consistent(col_info, row_info):
    """Sanity check: total filled cells implied by rows == by columns."""
    return sum(sum(c) for c in col_info) == sum(sum(r) for r in row_info)


def _repair(col, row, n, edits, budget, used=frozenset()):
    """Find a consistent, solvable reading by applying candidate repair edits
    (re-merges and digit swaps) likeliest-first, using the solver as oracle.
    Each edit must reduce the row/col sum imbalance (so the search terminates)
    and at most one edit applies per line. budget bounds the whole search.
    Returns (col, row, board) or None."""
    import solver2
    budget[0] -= 1
    if budget[0] <= 0:
        return None
    if clues_consistent(col, row):
        b = solver2.solve(col, row)
        return (col, row, b) if b else None

    def measure(c, r):                       # weighted error: feasibility first
        infeas = sum(1 for ln in c + r if sum(ln) + len(ln) - 1 > n)
        return 1000 * infeas + abs(sum(map(sum, c)) - sum(map(sum, r)))

    cur = measure(col, row)
    for _cost, side, li, new_line in edits:
        if (side, li) in used:
            continue
        if side == "row":
            ncol = col
            nrow = [new_line if k == li else l for k, l in enumerate(row)]
        else:
            ncol = [new_line if k == li else l for k, l in enumerate(col)]
            nrow = row
        # each edit must strictly reduce the weighted error (fixing an
        # infeasible over-merged line outweighs widening the imbalance). This
        # guarantees termination and keeps the search small.
        if measure(ncol, nrow) >= cur:
            continue
        res = _repair(ncol, nrow, n, edits, budget, used | {(side, li)})
        if res:
            return res
    return None


def solvable_clues(img_path, g, clf=None):
    """Read clues and return (col_info, row_info, board), repairing mis-split
    and uncertain-digit OCR errors using the solver as the oracle. Raises if no
    consistent, solvable reading is found."""
    col, row, edits = read_clues_full(img_path, g, clf)
    res = _repair(col, row, g["n"], edits[:20], budget=[6000])
    if res:
        return res
    raise AssertionError(
        f"unsolvable OCR (col_sum={sum(map(sum, col))} "
        f"row_sum={sum(map(sum, row))})")
