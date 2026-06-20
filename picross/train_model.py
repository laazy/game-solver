"""Append size-specific digit prototypes to clue_model.npz.

The classifier is multi-prototype: each digit may hold several templates and is
scored by its best match. A single mean template blurs small (30x30) digits --
notably '0', which appears only inside 10/20/30 and otherwise scores worst -- so
here we harvest real glyphs from a fully-known board and add a per-digit 30x30
mean prototype, leaving the existing templates untouched.

Ground truth comes from a board whose clue lines we know exactly (read once,
verified against the solver). For each clue line we segment its glyphs the same
way the OCR does, and only harvest when the segmentation matches the known
digit structure -- so a mis-split line is skipped, never mislabeled.

    python3 train_model.py        # harvest the live board + verify
"""
import numpy as np

import auto
import clue_ocr as C


# Verified ground truth for the on-screen 30x30 board (#1-48): every row clue
# read against the screenshot, every column from OCR with the one known fix
# (col 26 is [20], mis-read as [3,1]).  Rows alone are enough for digits 1-9;
# the 10/14/16/11/20 numbers supply the scarce '0'.
ROWS_GT = [
    [8], [5, 4], [8, 1, 4], [2, 4, 1, 1, 4], [2, 5, 1, 3, 6],
    [2, 3, 14], [4, 1, 4, 10], [3, 5, 10], [2, 2, 16], [2, 2, 1, 4, 11],
    [5, 1, 9, 5], [4, 8, 5], [3, 9, 4], [3, 9, 4], [4, 1, 2, 4, 4],
    [5, 1, 3, 1, 4], [1, 3, 1, 2, 4], [1, 1, 4, 4], [5, 8, 5], [6, 9, 5],
    [6, 10, 4], [5, 8, 1, 5], [3, 7, 4], [2, 6, 5], [2, 5, 5],
    [2, 2, 5], [2, 3, 5], [3, 2, 5], [4, 1, 4], [8],
]
COLS_GT = [
    [9], [8, 4], [11, 6], [3, 2, 11], [4, 2, 2, 5, 2], [2, 1, 3, 3, 2],
    [2, 1, 1, 2], [1, 1], [2, 2, 1, 2], [6, 1], [5, 1], [5, 1, 2],
    [5, 2, 1, 4, 1], [1, 1, 6, 2, 5, 1], [1, 6, 2, 8, 1], [1, 1, 6, 3, 8, 4],
    [1, 2, 7, 11, 1], [1, 3, 1, 4, 12, 1], [3, 13, 7, 2], [1, 11, 6, 2],
    [2, 11, 2, 3], [15, 4], [12, 1, 4], [11, 5], [7, 6], [8, 8],
    [20], [18], [14], [10],
]


def _lines(img, g):
    """Return per-line glyph groups for (cols, rows): each line is a list of
    numbers, each number a list of glyph dicts -- exactly the OCR grouping."""
    lu = C.luma_of(img)
    rgb = np.asarray(C.Image.open(img).convert("RGB")).astype(np.int32)
    n, cell = g["n"], g["cell"]
    top = C._drop_header(C.extract_glyphs(lu, int(g["x0"]), int(g["x1"]),
                                          0, int(g["y0"]), rgb=rgb))
    cols = [[] for _ in range(n)]
    for gl in top:
        j = int(round((gl["cx"] - g["x0"]) / cell - 0.5))
        if 0 <= j < n:
            cols[j].append(gl)
    left = C.extract_glyphs(lu, 0, int(g["x0"]), int(g["y0"]), int(g["y1"]),
                            rgb=rgb)
    rows = [[] for _ in range(n)]
    for gl in left:
        i = int(round((gl["cy"] - g["y0"]) / cell - 0.5))
        if 0 <= i < n:
            rows[i].append(gl)
    col_groups = [C._group_color(c, vertical=True) for c in cols]
    row_groups = [C._group_color(r) for r in rows]
    return col_groups, row_groups


def harvest(img, g, cols_gt, rows_gt):
    """(norm_bitmap, label) pairs from lines whose segmentation matches truth."""
    col_groups, row_groups = _lines(img, g)
    samples, skipped = [], 0
    for groups, truth in [(col_groups, cols_gt), (row_groups, rows_gt)]:
        for nums, vals in zip(groups, truth):
            if len(nums) != len(vals):
                skipped += 1
                continue
            for grp, v in zip(nums, vals):
                digs = [int(d) for d in str(v)]
                if len(grp) != len(digs):
                    skipped += 1
                    continue
                for gl, d in zip(sorted(grp, key=lambda d: d["cx"]), digs):
                    samples.append((gl["norm"], d))
    return samples, skipped


def main():
    img = auto.screenshot("/tmp/train.png")
    g = auto.detect_grid(img, n=None)
    assert g["n"] == 30, f"expected the 30x30 board, got n={g['n']}"
    samples, skipped = harvest(img, g, COLS_GT, ROWS_GT)
    by = {}
    for bmp, lab in samples:
        by.setdefault(lab, []).append(C._signed(bmp))
    print(f"harvested {len(samples)} glyphs ({skipped} lines skipped)")
    print("per digit:", {k: len(by[k]) for k in sorted(by)})

    # Add each 30x30 glyph as its own prototype (nearest-neighbour): a mean of
    # several differently-scaled samples blurs and still loses (a small '0'
    # then scores below '1'); an individual sample matches its twin exactly.
    keys, templates = C.load_model()
    new_keys = list(keys)
    new_tmpl = [t for t in templates]
    for d in sorted(by):
        for sig in by[d]:
            new_keys.append(d)
            new_tmpl.append(sig)
    C.save_model(np.array(new_keys), np.stack(new_tmpl))
    print(f"model now has {len(new_keys)} prototypes (was {len(keys)})")


if __name__ == "__main__":
    main()
