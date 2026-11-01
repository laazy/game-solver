"""solver3: pure line-propagation by row/column sweeps (no backtracking).

Implements the algorithm described as:
  1. N = 0, board state all 0 (unknown)
  2. For row N, enumerate every filling consistent with the current board;
     any cell that is filled (or empty) in *all* fillings is forced to 1 (or -1)
  3. N += 1
  4. when N reaches the row count: if the board is complete, stop;
     otherwise transpose the board (so the next sweep works on columns) and N = 0
  5. go back to step 2

Cells: 0 unknown, 1 filled, -1 empty.

This is the same line-intersection logic the older code used, driven by the
row/transpose sweep you described. It has no search fallback, so it solves
exactly the puzzles that are "line-solvable". A no-progress guard is added so
puzzles that need guessing terminate (returning the partial board) instead of
looping forever -- the literal algorithm in step 4 would spin indefinitely.
"""

from solver2 import line_options


def _transpose(board):
    return [list(row) for row in zip(*board)]


def _line_forced(clue, line):
    """Intersect all consistent fillings of `clue` into `line`.

    Returns the line with forced cells resolved, or None on contradiction."""
    opts = line_options(clue, line)
    if not opts:
        return None
    out = list(line)
    for j in range(len(line)):
        if out[j] != 0:
            continue
        v = opts[0][j]
        if all(o[j] == v for o in opts):
            out[j] = v
    return out


def _complete(board):
    return all(c != 0 for row in board for c in row)


def solve(col_info, row_info):
    """Return the solved board (1 filled / -1 empty), or the best partial board
    (still containing 0s) if line propagation alone can't finish it."""
    board = [[0] * len(col_info) for _ in range(len(row_info))]

    # `clues` describes the rows of the board in its current orientation.
    # board starts as row_info x col_info; after a transpose the roles swap.
    clues = row_info
    other = col_info
    transposed = False

    stalled = 0  # consecutive sweeps that changed nothing
    while True:
        changed = False
        for n in range(len(clues)):          # steps 2-3: sweep every row
            res = _line_forced(clues[n], board[n])
            if res is None:
                return None                  # contradiction -> unsatisfiable
            if res != board[n]:
                board[n] = res
                changed = True

        # step 4: finished a full orientation
        if _complete(board):
            return board if not transposed else _transpose(board)

        stalled = 0 if changed else stalled + 1
        if stalled >= 2:                     # a row-pass and a col-pass both stuck
            return board if not transposed else _transpose(board)

        board = _transpose(board)            # switch to the other orientation
        clues, other = other, clues
        transposed = not transposed
