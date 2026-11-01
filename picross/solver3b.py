"""solver3b: same pure line-propagation as solver3, but lines are processed in
a global priority order -- most-constrained first.

Priority = sum(clue) + len(clue).  The larger it is, the smaller the slack
  slack = width + 1 - (sum + len)
so the fewer fillings line_options() must enumerate, AND the more cells the
line forces.  Doing the tight lines first fills cells that shrink the loose
lines before we ever enumerate them.

Rows and columns are mixed in one queue (no transpose sweeps), so a tight
column can run before a loose row.
"""
from solver2 import line_options

# instrumentation: total candidate tuples line_options produced this solve
COUNT = {"opts": 0, "calls": 0}


def _line_forced(clue, line):
    opts = line_options(clue, line)
    COUNT["opts"] += len(opts)
    COUNT["calls"] += 1
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


def solve(col_info, row_info, ordered=True):
    n, m = len(row_info), len(col_info)
    board = [[0] * m for _ in range(n)]

    # build the line list: ('r', i) and ('c', j), with priorities
    lines = ([("r", i, sum(row_info[i]) + len(row_info[i])) for i in range(n)]
             + [("c", j, sum(col_info[j]) + len(col_info[j])) for j in range(m)])
    if ordered:
        lines.sort(key=lambda x: -x[2])           # most-constrained first

    def get_line(kind, idx):
        if kind == "r":
            return board[idx]
        return [board[i][idx] for i in range(n)]

    def put_line(kind, idx, res):
        if kind == "r":
            board[idx] = res
        else:
            for i in range(n):
                board[i][idx] = res[i]

    stalled_passes = 0
    while True:
        changed = False
        for kind, idx, _ in lines:
            clue = row_info[idx] if kind == "r" else col_info[idx]
            cur = get_line(kind, idx)
            if all(c != 0 for c in cur):          # already solved -> skip
                continue
            res = _line_forced(clue, cur)
            if res is None:
                return None
            if res != cur:
                put_line(kind, idx, res)
                changed = True
        if all(c != 0 for r in board for c in r):
            return board
        if not changed:
            return board                          # stuck (needs guessing)
