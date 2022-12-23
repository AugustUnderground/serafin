import time
import serafin as sf
import pyspectre as ps
import numpy as np
import pandas as pd
from functools import partial
from concurrent.futures import ThreadPoolExecutor

# MAKRE SURE `include: path` is updated in `example/gpdk180.yml`

pdk = './example/gpdk180.yml'
net = './example/sym.scs'
ckt = './example/sym.yml'
num = 10

## Single
sym  = sf.operational_amplifier(pdk, ckt, net)

size = sf.random_sizing(sym)
prf  = sf.evaluate(sym,size)

def test_perf(i, o):
    tic  = time.time()
    p    = sf.current_sizing(o)
    s    = sf.random_sizing(o)
    prf  = sf.evaluate(o,s)
    n    = sf.current_sizing(o)
    d    = s.loc[:,((s != n).values)[0].tolist()] \
         - n.loc[:,((s != n).values)[0].tolist()]
    toc  = time.time()
    took = toc - tic
    print(f'Iteration {i} took {took:.3}s')
    #print(d)
    return took

times = np.array([ test_perf(i, sym) for i in range(24) ])
print(f'Average: {np.mean(times):.3}s')

## Parallel
tic = time.time()
with ThreadPoolExecutor(max_workers = num) as tpe:
    args  = zip(num * [pdk], num * [ckt], num * [net])
    syms  = list(tpe.map(lambda a: sf.operational_amplifier(*a), args))
toc = time.time()
print(f'Creating took {toc - tic}s')

def test_perf_parallel(i,tpe):
    pf = list(tpe.map(partial(test_perf, i), syms))
    return pf

with ThreadPoolExecutor(max_workers = num) as tpe:
    times = np.array([ test_perf_parallel(i,tpe) for i in range(50) ])

print(f'Average: {np.mean(times):.3}s')
