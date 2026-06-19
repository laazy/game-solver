"""Robust nonogram solver: line propagation to a fixpoint, then backtracking.

The original picross_solver does line propagation well but its guessing
fallback is unreliable, so puzzles that aren't fully line-solvable (e.g. ones
with long runs / empty lines) can fail. This solver wraps the same line logic
in proper recursive search and always finds a solution if one exists.

Cells: 0 unknown, 1 filled, -1 empty.
"""
import copy


def _consistent(cand, known):
    return all(k == 0 or k == c for c, k in zip(cand, known))


def line_options(clue, line):
    """All fillings of `clue` into a line of len(line), consistent with the
    already-known cells (0 unknown / 1 / -1)."""
    n = len(line)
    if not clue or clue == [0]:
        cand = [-1] * n
        return [tuple(cand)] if _consistent(cand, line) else []
    res = []

    def rec(idx, cur):
        b = clue[idx]
        remain = sum(clue[idx + 1:]) + (len(clue) - idx - 1)
        for start in range(len(cur), n - b - remain + 1):
            cand = cur + [-1] * (start - len(cur)) + [1] * b
            if not _consistent(cand, line[:len(cand)]):
                continue
            if idx == len(clue) - 1:
                full = cand + [-1] * (n - len(cand))
                if _consistent(full, line):
                    res.append(tuple(full))
            else:
                rec(idx + 1, cand + [-1])

    rec(0, [])
    return res


def _propagate(board, row_info, col_info):
    """Fill in all forced cells. Returns False on contradiction."""
    n, m = len(row_info), len(col_info)
    changed = True
    while changed:
        changed = False
        for i in range(n):
            opts = line_options(row_info[i], board[i])
            if not opts:
                return False
            for j in range(m):
                if board[i][j] == 0:
                    vals = {o[j] for o in opts}
                    if len(vals) == 1:
                        board[i][j] = vals.pop()
                        changed = True
        for j in range(m):
            col = [board[i][j] for i in range(n)]
            opts = line_options(col_info[j], col)
            if not opts:
                return False
            for i in range(n):
                if board[i][j] == 0:
                    vals = {o[i] for o in opts}
                    if len(vals) == 1:
                        board[i][j] = vals.pop()
                        changed = True
    return True


def _search(board, row_info, col_info):
    b = copy.deepcopy(board)
    if not _propagate(b, row_info, col_info):
        return None
    spot = next(((i, j) for i in range(len(row_info))
                 for j in range(len(col_info)) if b[i][j] == 0), None)
    if spot is None:
        return b
    i, j = spot
    for val in (1, -1):
        b[i][j] = val
        r = _search(b, row_info, col_info)
        if r:
            return r
        b[i][j] = 0
    return None


def solve(col_info, row_info):
    """Return the solved board (1 filled / -1 empty) or None if unsatisfiable."""
    board = [[0] * len(col_info) for _ in range(len(row_info))]
    return _search(board, row_info, col_info)
