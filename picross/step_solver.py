"""Interactive step-through of the line-propagation solver.

Runs one orientation at a time: a full ROW pass, then waits for you to press
Enter, then a full COLUMN pass, waits again, and so on -- so you can watch the
deductions cascade. Cells resolved in the most recent pass are highlighted.

Usage:
    python3 step_solver.py [puzzle_path]      # default: puzzles/1-93.txt

At each prompt:  Enter = run next pass   |   a = auto-run to the end   |   q = quit
"""
import sys, json
from solver2 import line_options


def parse_txt(p):
    with open(p) as f:
        f.readline()                              # "rows cols" header (unused)
        col_line = f.readline().strip()
        row_line = f.readline().strip()
    col = [[int(x) for x in g.split()] for g in col_line.split(',')]
    row = [[int(x) for x in g.split()] for g in row_line.split(',')]
    return col, row


def load(p):
    if p.endswith('.json'):
        d = json.load(open(p))
        return d['col'], d['row']
    return parse_txt(p)


def line_forced(clue, line):
    """Forced cells from intersecting all consistent fillings; None on clash."""
    opts = line_options(clue, line)
    if not opts:
        return None
    out = list(line)
    for j in range(len(line)):
        if out[j] == 0 and all(o[j] == opts[0][j] for o in opts):
            out[j] = opts[0][j]
    return out


# ANSI colors
G, R, DIM, BOLD, RST = '\033[32m', '\033[31m', '\033[2m', '\033[1m', '\033[0m'


def render(board, fresh):
    """fresh: set of (i,j) resolved in the most recent pass -> highlighted."""
    n, m = len(board), len(board[0])
    head = '   ' + ''.join(str(j % 10) for j in range(m))
    lines = [head]
    for i, row in enumerate(board):
        cells = []
        for j, c in enumerate(row):
            ch = '#' if c == 1 else ('.' if c == -1 else ' ')
            if (i, j) in fresh:
                col = G if c == 1 else R
                cells.append(f'{BOLD}{col}{ch}{RST}')
            elif c == 0:
                cells.append(f'{DIM}·{RST}')
            else:
                cells.append(ch)
        lines.append(f'{i % 10:2d}|' + ''.join(cells) + '|')
    return '\n'.join(lines)


def run_pass(board, clues, kind):
    """Run a full pass over rows (kind='row') or columns (kind='col').
    Mutates board. Returns (newly_resolved set, ok)."""
    n, m = len(board), len(board[0])
    fresh = set()
    if kind == 'row':
        for i in range(n):
            res = line_forced(clues[i], board[i])
            if res is None:
                return fresh, False
            for j in range(m):
                if board[i][j] == 0 and res[j] != 0:
                    board[i][j] = res[j]
                    fresh.add((i, j))
    else:
        for j in range(m):
            col = [board[i][j] for i in range(n)]
            res = line_forced(clues[j], col)
            if res is None:
                return fresh, False
            for i in range(n):
                if board[i][j] == 0 and res[i] != 0:
                    board[i][j] = res[i]
                    fresh.add((i, j))
    return fresh, True


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'puzzles/1-93.txt'
    col_info, row_info = load(path)
    n, m = len(row_info), len(col_info)
    board = [[0] * m for _ in range(n)]

    print(f"{BOLD}{path}{RST}  {n}x{m}   (# filled, . empty, · unknown; "
          f"{G}green{RST}/{R}red{RST} = resolved this pass)\n")
    print(render(board, set()))

    passes = 0
    last_total = 0           # filled+empty count after previous pass
    stall = 0
    auto = False
    kinds = ('row', 'col')
    while True:
        kind = kinds[passes % 2]
        if not auto:
            cmd = input(f"\n[{passes+1}] press Enter to run {BOLD}{kind.upper()}"
                        f"{RST} pass  (a=auto, q=quit) > ").strip().lower()
            if cmd == 'q':
                print("bye")
                return
            if cmd == 'a':
                auto = True

        fresh, ok = run_pass(board, row_info if kind == 'row' else col_info, kind)
        passes += 1

        total = sum(1 for r in board for c in r if c != 0)
        unknown = n * m - total
        print(f"\n--- pass {passes} ({kind}) : resolved {len(fresh)} cells, "
              f"{unknown} unknown left ---")
        print(render(board, fresh))

        if not ok:
            print(f"\n{R}{BOLD}CONTRADICTION{RST} -- puzzle unsatisfiable.")
            return
        if unknown == 0:
            print(f"\n{G}{BOLD}SOLVED{RST} in {passes} passes.")
            return
        # stall = two consecutive passes (one row, one col) with no change
        stall = stall + 1 if total == last_total else 0
        last_total = total
        if stall >= 2:
            print(f"\n{R}{BOLD}STALLED{RST} -- line propagation can't resolve the "
                  f"remaining {unknown} cells (needs a guess).")
            return


if __name__ == '__main__':
    main()
