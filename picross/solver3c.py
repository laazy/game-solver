"""solver3c: ordered line-propagation solver (no backtracking).

  * DP feasibility (solver2._line_forced) instead of enumerating fillings, so a
    wide, lightly-constrained line is cheap rather than blowing up to hundreds
    of thousands of candidates (the failure mode in solver3/solver3b).
  * sum(clue)+len(clue) ordering heuristic -- process the most-constrained
    lines first. Measured ~10x fewer line evaluations than index order, because
    tight lines are cheap and pin cells that shrink the loose lines.

Pure line propagation only -- the backtracking guess fallback was removed. So
solve() finishes any line-solvable puzzle but, on a puzzle that needs a guess
(e.g. 1-93), returns a PARTIAL board with unknown cells left as 0 rather than
guessing. Returns None only when the clues are contradictory.
"""
from solver2 import _line_forced     # DP-based forced-cell resolver (no enum)


def _order(info):
    """Line indices most-constrained first (largest sum+blocks == least slack)."""
    return sorted(range(len(info)), key=lambda i: -(sum(info[i]) + len(info[i])))


def _propagate(board, row_info, col_info, row_ord, col_ord):
    """Fill all forced cells to a fixpoint, tight lines first. False on clash."""
    n, m = len(row_info), len(col_info)
    changed = True
    while changed:
        changed = False
        for i in row_ord:
            res = _line_forced(row_info[i], board[i])
            if res is None:
                return False
            for j in range(m):
                if board[i][j] == 0 and res[j] != 0:
                    board[i][j] = res[j]
                    changed = True
        for j in col_ord:
            col = [board[i][j] for i in range(n)]
            res = _line_forced(col_info[j], col)
            if res is None:
                return False
            for i in range(n):
                if board[i][j] == 0 and res[i] != 0:
                    board[i][j] = res[i]
                    changed = True
    return True


def solve(col_info, row_info):
    """Run ordered line propagation to a fixpoint.

    Returns the board (1 filled / -1 empty, 0 still-unknown for the cells line
    propagation alone can't decide), or None if the clues are contradictory.
    No backtracking -- a guess-requiring puzzle comes back partially solved."""
    row_ord = _order(row_info)
    col_ord = _order(col_info)
    board = [[0] * len(col_info) for _ in range(len(row_info))]
    if not _propagate(board, row_info, col_info, row_ord, col_ord):
        return None
    return board
