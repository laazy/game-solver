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


def _feasible(clue, line):
    """Can `clue` (no zeros) be packed into `line` (0 unknown / 1 / -1)?

    O(n*k) memoized DP over (cell index, block index) -- never enumerates the
    arrangements, so it does not blow up on wide, lightly-constrained lines the
    way line_options() does."""
    n, k = len(line), len(clue)
    memo = {}

    def go(i, b):
        if i >= n:
            return b == k
        key = (i, b)
        if key in memo:
            return memo[key]
        res = False
        if line[i] != 1:                       # leave cell i empty
            res = go(i + 1, b)
        if not res and b < k:                  # place block b starting at i
            L = clue[b]
            end = i + L
            if end <= n and all(line[x] != -1 for x in range(i, end)):
                if end == n:
                    res = (b + 1 == k)
                elif line[end] != 1:           # mandatory gap cell after a block
                    res = go(end + 1, b + 1)
        memo[key] = res
        return res

    return go(0, 0)


def _line_forced(clue, line):
    """Return `line` with every forced cell resolved (0/1/-1), or None if the
    clue can't fit. A cell is forced when only one value keeps the line
    feasible -- decided by two feasibility probes per unknown cell, no
    enumeration."""
    clue = [c for c in clue if c]              # drop the empty-line [0] marker
    n = len(line)
    if not _feasible(clue, line):
        return None
    out = list(line)
    for j in range(n):
        if out[j] != 0:
            continue
        t = list(line)
        t[j] = 1
        can_fill = _feasible(clue, t)
        t[j] = -1
        can_empty = _feasible(clue, t)
        if can_fill and not can_empty:
            out[j] = 1
        elif can_empty and not can_fill:
            out[j] = -1
        elif not can_fill and not can_empty:
            return None
    return out


def _propagate(board, row_info, col_info):
    """Fill in all forced cells. Returns False on contradiction."""
    n, m = len(row_info), len(col_info)
    changed = True
    while changed:
        changed = False
        for i in range(n):
            res = _line_forced(row_info[i], board[i])
            if res is None:
                return False
            for j in range(m):
                if board[i][j] == 0 and res[j] != 0:
                    board[i][j] = res[j]
                    changed = True
        for j in range(m):
            col = [board[i][j] for i in range(n)]
            res = _line_forced(col_info[j], col)
            if res is None:
                return False
            for i in range(n):
                if board[i][j] == 0 and res[i] != 0:
                    board[i][j] = res[i]
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
