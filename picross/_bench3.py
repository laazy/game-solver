"""Compare solver3 (line-only sweeps) vs solver2 (line + backtracking)."""
import json, os, time, glob, signal
import solver2, solver3


class Timeout(Exception):
    pass


def run(fn, *a, limit=15):
    def _h(*_):
        raise Timeout
    old = signal.signal(signal.SIGALRM, _h)
    signal.setitimer(signal.ITIMER_REAL, limit)
    t = time.perf_counter()
    try:
        b = fn(*a)
        return b, time.perf_counter() - t, False
    except Timeout:
        return None, limit, True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def parse_txt(path):
    with open(path) as f:
        first = f.readline().split()
        rows_n, cols_n = int(first[0]), int(first[1])
        col_line = f.readline().strip()
        row_line = f.readline().strip()
    col = [[int(x) for x in g.split()] for g in col_line.split(',')]
    row = [[int(x) for x in g.split()] for g in row_line.split(',')]
    return col, row


def parse_json(path):
    d = json.load(open(path))
    return d['col'], d['row']


def load(path):
    return parse_json(path) if path.endswith('.json') else parse_txt(path)


def complete(b):
    return b is not None and all(c != 0 for row in b for c in row)


for path in sorted(glob.glob('puzzles/*')):
    try:
        col, row = load(path)
    except Exception as e:
        print(f"{os.path.basename(path):12} SKIP ({e})")
        continue
    size = f"{len(row)}x{len(col)}"

    b2, t2, to2 = run(solver2.solve, col, row)
    b3, t3, to3 = run(solver3.solve, col, row)

    def stat(b, to):
        if to:
            return "TIMEOUT"
        return "solved" if complete(b) else ("partial" if b is not None else "none")
    s2, s3 = stat(b2, to2), stat(b3, to3)
    match = "  <-- DIFFER" if complete(b2) and complete(b3) and b2 != b3 else ""
    print(f"{os.path.basename(path):12} {size:7} "
          f"solver2={s2:7}({t2*1000:7.1f}ms)  "
          f"solver3={s3:7}({t3*1000:7.1f}ms){match}")
