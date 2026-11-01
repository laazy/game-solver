import time, auto, clue_ocr as C, solver2
img='/tmp/play.png'
g=auto.detect_grid(img,n=None)
cols,rows=C.read_clues(img,g)
cs,rs=sum(map(sum,cols)),sum(map(sum,rows))
print('col_sum',cs,'row_sum',rs)

def inc(line):  # +1 variants: bump a digit, or insert a dropped 1
    out=[]
    for i in range(len(line)):
        c=list(line); c[i]+=1; out.append(c)
    for i in range(len(line)+1):
        out.append(line[:i]+[1]+line[i:])
    return out
def dec(line):  # -1 variants: drop a digit by 1, or remove a lone 1
    out=[]
    for i in range(len(line)):
        if line[i]>1: c=list(line); c[i]-=1; out.append(c)
        elif line[i]==1 and len(line)>1: out.append(line[:i]+line[i+1:])
    return out

t0=time.time(); hits=[]
# Case A: one column read low (+1)
for j in range(len(cols)):
    for cand in inc(cols[j]):
        tr=[list(c) for c in cols]; tr[j]=cand
        if sum(map(sum,tr))!=rs: continue
        if solver2.solve(tr,rows) is not None:
            hits.append(('COL+',j,cols[j],cand)); print('FIX col',j,cols[j],'->',cand)
# Case B: one row read high (-1)
for i in range(len(rows)):
    for cand in dec(rows[i]):
        tr=[list(r) for r in rows]; tr[i]=cand
        if sum(map(sum,tr))!=cs: continue
        if solver2.solve(cols,tr) is not None:
            hits.append(('ROW-',i,rows[i],cand)); print('FIX row',i,rows[i],'->',cand)
print('hits',len(hits),'%.0fs'%(time.time()-t0))
