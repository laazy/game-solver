import solver3b

def parse_txt(p):
    with open(p) as f:
        a = f.readline().split(); c = f.readline().strip(); r = f.readline().strip()
    col = [[int(x) for x in g.split()] for g in c.split(',')]
    row = [[int(x) for x in g.split()] for g in r.split(',')]
    return col, row

col, row = parse_txt('puzzles/1-93.txt')
b = solver3b.solve(col, row, ordered=True)

GLYPH = {1: '#', -1: '.', 0: '?'}
unknown = sum(1 for r in b for c in r if c == 0)
total = len(b) * len(b[0])
print(f"stall board for 1-93  ({len(b)}x{len(b[0])})  "
      f"unknown {unknown}/{total} ({100*unknown/total:.0f}%)\n")
for r in b:
    print(''.join(GLYPH[c] for c in r))
