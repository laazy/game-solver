"""Profile which line_options() calls dominate solver3 on 1-124."""
import json, time, math
import solver2, solver3

d = json.load(open('puzzles/1-124.json'))
col, row = d['col'], d['row']

stats = []  # (elapsed, n_options, clue, line)
_orig = solver2.line_options


def traced(clue, line):
    t = time.perf_counter()
    res = _orig(clue, line)
    stats.append((time.perf_counter() - t, len(res), tuple(clue), tuple(line)))
    return res


solver2.line_options = traced
solver3.line_options = traced

t = time.perf_counter()
b = solver3.solve(col, row)
total = time.perf_counter() - t

print(f"total solve: {total:.2f}s   line_options calls: {len(stats)}")

# theoretical arrangement count for a clue in a width-w EMPTY line:
def arrangements(clue, w):
    k = len(clue)
    slack = w - (sum(clue) + k - 1)
    if slack < 0:
        return 0
    return math.comb(slack + k, k)

print("\nslowest single calls:")
for el, no, clue, line in sorted(stats, reverse=True)[:8]:
    known = sum(1 for c in line if c != 0)
    print(f"  {el*1000:7.1f}ms  opts={no:7d}  known={known:2d}/{len(line)}  "
          f"clue={list(clue)}  arrange(empty)={arrangements(clue, len(line))}")

# aggregate by clue (a "line" = a fixed row/col clue, called many times)
agg = {}
for el, no, clue, line in stats:
    a = agg.setdefault(clue, [0.0, 0, 0])
    a[0] += el; a[1] += 1; a[2] = max(a[2], no)
print("\nclues by total time spent:")
for clue, (tot, cnt, mx) in sorted(agg.items(), key=lambda x: -x[1][0])[:8]:
    print(f"  {tot*1000:8.1f}ms over {cnt:4d} calls  max_opts={mx:8d}  "
          f"clue={list(clue)}  sum={sum(clue)}  blocks={len(clue)}")
