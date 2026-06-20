"""Harvest 30x30 row-clue glyphs from board #1-124 and add them as prototypes.

The row clues were verified by eye against the screenshot (all match the OCR
except row 19, whose leading digit is 8, mis-read as 3). So the rows are
ground truth -- we harvest only the rows (columns still hold the error we're
chasing) and add each glyph as a 30x30 prototype, raising the margin on the
small-digit confusions (notably 3 vs 8) that break dense 30x30 boards.

    python3 harvest_124.py        # extends clue_model.npz
"""
import numpy as np

import auto
import clue_ocr as C
import train_model as T

IMG = "/tmp/a1.png"

# Verified by vision from the screenshot (rows 0..29). Identical to the OCR
# read except row 19 (leading 8, not 3).
ROWS_GT = [
    [7, 2], [5, 13], [4, 2, 2, 2], [1, 2, 2, 2, 2], [1, 2, 1, 1, 1],
    [4, 17], [1, 1, 1, 5, 1, 3, 1], [1, 1, 1, 5, 1, 3, 1, 1],
    [1, 1, 8, 1, 3, 1, 3], [6, 1, 8, 5], [1, 1, 2, 2, 5],
    [1, 2, 1, 1, 3, 1], [1, 2, 3, 1, 3, 1, 2, 1], [1, 1, 1, 3, 2, 1, 2, 1],
    [1, 4, 1, 1, 1, 3, 1, 4], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 3, 1, 3, 1, 1, 1], [1, 5, 1, 1], [5, 19, 1], [8, 2, 3, 2, 1],
    [8, 4, 3, 4, 4], [1, 2, 1, 1, 1, 2, 2], [1, 1, 1, 1, 2, 11, 1, 2],
    [1, 1, 1, 1, 2, 1, 1, 1, 1, 5], [1, 1, 1, 1, 2, 1, 7, 1, 3],
    [1, 2, 2, 1, 2, 2, 3, 1], [11, 1, 1, 1, 1], [4, 20, 1],
    [6, 1, 3, 4, 4, 2], [3, 5, 3, 4, 4, 1],
]


def main():
    g = auto.detect_grid(IMG, n=None)
    assert g["n"] == 30, g["n"]
    _, row_groups = T._lines(IMG, g)
    samples, skipped = [], 0
    for nums, vals in zip(row_groups, ROWS_GT):
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
    by = {}
    for bmp, lab in samples:
        by.setdefault(lab, []).append(C._signed(bmp))
    print(f"harvested {len(samples)} row glyphs ({skipped} lines skipped)")
    print("per digit:", {k: len(by[k]) for k in sorted(by)})

    keys, templates = C.load_model()
    new_keys, new_tmpl = list(keys), [t for t in templates]
    for d in sorted(by):
        for sig in by[d]:
            new_keys.append(d)
            new_tmpl.append(sig)
    C.save_model(np.array(new_keys), np.stack(new_tmpl))
    print(f"model now has {len(new_keys)} prototypes (was {len(keys)})")


if __name__ == "__main__":
    main()
