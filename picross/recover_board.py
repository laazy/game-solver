"""Recover the true 30x30 board by jointly freeing the misread lines.

We do NOT trust any single clue line (eye-verification of a 30x30 is itself
unreliable). Instead: the misread lines are exactly the ones whose constraints,
when dropped, let the *remaining* rows+columns determine a board. We free the
smallest set of lines (rows or columns, in any mix) that yields a solution,
then read the TRUE clue off the recovered board. A consistent unique board is
ground truth regardless of which glyphs the OCR got wrong.

    python3 recover_board.py            # uses /tmp/play.png
"""
import copy
import itertools

import auto
import clue_ocr as C
import solver2

IMG = "/tmp/play.png"


def _propagate(board, row_info, col_info, free_rows, free_cols):
    n, m = len(row_info), len(col_info)
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if i in free_rows:
                continue
            res = solver2._line_forced(row_info[i], board[i])
            if res is None:
                return False
            for j in range(m):
                if board[i][j] == 0 and res[j] != 0:
                    board[i][j] = res[j]; changed = True
        for j in range(m):
            if j in free_cols:
                continue
            col = [board[i][j] for i in range(n)]
            res = solver2._line_forced(col_info[j], col)
            if res is None:
                return False
            for i in range(n):
                if board[i][j] == 0 and res[i] != 0:
                    board[i][j] = res[i]; changed = True
    return True


def _search(board, row_info, col_info, free_rows, free_cols):
    b = copy.deepcopy(board)
    if not _propagate(b, row_info, col_info, free_rows, free_cols):
        return None
    spot = next(((i, j) for i in range(len(row_info))
                 for j in range(len(col_info)) if b[i][j] == 0), None)
    if spot is None:
        return b
    i, j = spot
    for val in (1, -1):
        b[i][j] = val
        r = _search(b, row_info, col_info, free_rows, free_cols)
        if r:
            return r
        b[i][j] = 0
    return None


def solve_free(col_info, row_info, free_rows, free_cols):
    board = [[0] * len(col_info) for _ in range(len(row_info))]
    return _search(board, row_info, col_info,
                   frozenset(free_rows), frozenset(free_cols))


def _runs(line_vals):
    runs, c = [], 0
    for v in line_vals:
        if v == 1:
            c += 1
        elif c:
            runs.append(c); c = 0
    if c:
        runs.append(c)
    return runs or [0]


def col_clue(b, j):
    return _runs([b[i][j] for i in range(len(b))])


def row_clue(b, i):
    return _runs(b[i])


def _minlen(clue):
    """Minimum cells a clue needs: blocks + one gap between each."""
    cl = [c for c in clue if c]
    return sum(cl) + max(0, len(cl) - 1)


def recover(cols, rows, max_free=2, verbose=False):
    """Find the true board when a few clue lines are misread.

    Structurally-impossible lines (minlen > n, e.g. a phantom edge block making
    [30,1] on a 30-wide row) are CERTAINLY wrong, so they're freed for free --
    no search cost. We then search up to max_free *additional* lines (rows or
    cols, any mix); the remaining lines determine the true board and we read the
    fixed clue off it. Error-type agnostic: phantom/dropped block or digit swap.
    Returns (cols_fixed, rows_fixed, board) or None.
    """
    n = len(rows)
    # Repair forced-full lines: a clue whose largest block == n means the line
    # is entirely filled, so it can only be [n] -- pin it (dropping phantom edge
    # blocks like [30,1] -> [30]). Pinning, not freeing, keeps the board
    # constrained so the solver can't fabricate an alternative.
    cols = [[n] if (_minlen(c) > n and max(c) >= n) else c for c in cols]
    rows = [[n] if (_minlen(r) > n and max(r) >= n) else r for r in rows]
    # Any still-impossible line is ambiguous -> free it (no search cost).
    forced_fr = {i for i in range(n) if _minlen(rows[i]) > n}
    forced_fc = {j for j in range(len(cols)) if _minlen(cols[j]) > n}
    if not forced_fr and not forced_fc:
        sol = solver2.solve(cols, rows)
        if sol is not None:
            return cols, rows, sol
    rem = ([("r", i) for i in range(n) if i not in forced_fr] +
           [("c", j) for j in range(len(cols)) if j not in forced_fc])
    for k in range(0, max_free + 1):       # k=0 = free only the impossible lines
        if verbose:
            print(f"-- free {len(forced_fr)+len(forced_fc)}+{k} line(s) --",
                  flush=True)
        for combo in itertools.combinations(rem, k):
            fr = forced_fr | {i for t, i in combo if t == "r"}
            fc = forced_fc | {j for t, j in combo if t == "c"}
            sol = solve_free(cols, rows, fr, fc)
            if sol is None:
                continue
            # searched (non-forced) frees must come back changed -- else they
            # weren't really errors (avoids spurious "fixes").
            srch_r = {i for t, i in combo if t == "r"}
            srch_c = {j for t, j in combo if t == "c"}
            if all(row_clue(sol, i) == rows[i] for i in srch_r) and \
               all(col_clue(sol, j) == cols[j] for j in srch_c):
                continue
            fixed_c = [col_clue(sol, j) if j in fc else cols[j]
                       for j in range(len(cols))]
            fixed_r = [row_clue(sol, i) if i in fr else rows[i]
                       for i in range(len(rows))]
            return fixed_c, fixed_r, sol
    return None


def main():
    g = auto.detect_grid(IMG, n=None)
    assert g["n"] == 30, g["n"]
    cols, rows = C.read_clues(IMG, g)
    print("col_sum", sum(map(sum, cols)), "row_sum", sum(map(sum, rows)),
          flush=True)
    out = recover(cols, rows, max_free=2, verbose=True)
    if out is None:
        print("no solution within 2 freed lines", flush=True)
        return
    fixed_c, fixed_r, sol = out
    for i in range(len(rows)):
        if fixed_r[i] != rows[i]:
            print("  row %2d: OCR %s -> TRUE %s" % (i, rows[i], fixed_r[i]))
    for j in range(len(cols)):
        if fixed_c[j] != cols[j]:
            print("  col %2d: OCR %s -> TRUE %s" % (j, cols[j], fixed_c[j]))
    with open("/tmp/board_124.txt", "w") as f:
        f.write("COLS=%r\nROWS=%r\n" % (fixed_c, fixed_r))
        for r in sol:
            f.write("".join("#" if v == 1 else "." for v in r) + "\n")
    print("  saved -> /tmp/board_124.txt", flush=True)


if __name__ == "__main__":
    main()
