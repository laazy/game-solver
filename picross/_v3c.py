import json, glob, os, time, signal
import solver2, solver3c
class TO(Exception): pass
def run(fn,limit=20):
    def h(*_): raise TO
    signal.signal(signal.SIGALRM,h); signal.setitimer(signal.ITIMER_REAL,limit)
    t=time.perf_counter()
    try: return fn(), time.perf_counter()-t, False
    except TO: return None, limit, True
    finally: signal.setitimer(signal.ITIMER_REAL,0)
def load(p):
    if p.endswith('.json'):
        d=json.load(open(p)); return d['col'],d['row']
    with open(p) as f:
        f.readline(); c=f.readline().strip(); r=f.readline().strip()
    return ([[int(x) for x in g.split()] for g in c.split(',')],
            [[int(x) for x in g.split()] for g in r.split(',')])
def done(b): return b is not None and all(c!=0 for r in b for c in r)
for p in sorted(glob.glob('puzzles/*')):
    try: col,row=load(p)
    except Exception: continue
    b2,t2,to2=run(lambda:solver2.solve(col,row))
    bc,tc,toc=run(lambda:solver3c.solve(col,row))
    same = (b2==bc)
    def st(b,to): return 'TIMEOUT' if to else ('solved' if done(b) else ('none' if b is None else 'partial'))
    print(f"{os.path.basename(p):12} {len(row)}x{len(col)}  "
          f"solver2={st(b2,to2):7}{t2*1000:7.0f}ms | "
          f"solver3c={st(bc,toc):7}{tc*1000:7.0f}ms | match={same}")
