"""Measure index-order vs constrained-first ordering on the same puzzles."""
import json, time, glob, os, signal
import solver3b


class TO(Exception): pass
def run(fn, limit=30):
    def h(*_): raise TO
    signal.signal(signal.SIGALRM, h); signal.setitimer(signal.ITIMER_REAL, limit)
    t = time.perf_counter()
    try:
        return fn(), time.perf_counter() - t, False
    except TO:
        return None, limit, True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def parse_txt(p):
    with open(p) as f:
        a = f.readline().split(); c = f.readline().strip(); r = f.readline().strip()
    col = [[int(x) for x in g.split()] for g in c.split(',')]
    row = [[int(x) for x in g.split()] for g in r.split(',')]
    return col, row
def load(p):
    if p.endswith('.json'):
        d = json.load(open(p)); return d['col'], d['row']
    return parse_txt(p)
def complete(b): return b is not None and all(c != 0 for r in b for c in r)

for path in sorted(glob.glob('puzzles/*')):
    try: col, row = load(path)
    except Exception: continue
    name = os.path.basename(path)

    solver3b.COUNT.update(opts=0, calls=0)
    b0, t0, to0 = run(lambda: solver3b.solve(col, row, ordered=False))
    o0, k0 = solver3b.COUNT["opts"], solver3b.COUNT["calls"]

    solver3b.COUNT.update(opts=0, calls=0)
    b1, t1, to1 = run(lambda: solver3b.solve(col, row, ordered=True))
    o1, k1 = solver3b.COUNT["opts"], solver3b.COUNT["calls"]

    def f(b, to): return "TIMEOUT" if to else ("solved" if complete(b) else ("partial" if b else "none"))
    sp = f"{t0/t1:.1f}x" if (not to0 and not to1 and t1 > 0) else "-"
    print(f"{name:12} {len(row)}x{len(col)}  "
          f"index:{f(b0,to0):7}{t0*1000:7.0f}ms opts={o0:>9}  | "
          f"sorted:{f(b1,to1):7}{t1*1000:7.0f}ms opts={o1:>9}  | speedup {sp}")
