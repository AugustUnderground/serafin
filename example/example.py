import time
import serafin as sf
import pyspectre as ps
import numpy as np
import pandas as pd
from multiprocessing.dummy import Pool

# MAKRE SURE `include: path` is updated in `example/gpdk180.yml`

pdk = './example/gpdk090.yml'
net = './example/sym.scs'
ckt = './example/sym.yml'
num = 10

## Single
sym = sf.operational_amplifier(pdk, ckt, net)
sf.current_sizing(sym)
size = sf.random_sizing(sym)
prf = sf.evaluate(sym,size)

def test_perf(i):
    tic = time.time()
    sz  = sf.random_sizing(sym)
    prf = sf.evaluate(sym,sz)
    toc = time.time()
    took = toc - tic
    print(f'Iteration {i} took {took}s')
    return took

times = np.array([ test_perf(i) for i in range(10) ])
print(f'Average: {np.mean(times):.3}s')

## Parallel

sizes_df = pd.DataFrame()
perfs_df = pd.DataFrame()

def test_perf_parallel(op, numWorkers):
    tic = time.time()
    with Pool(numWorkers) as pl:
        sz = pl.map(sf.random_sizing, op)
        prf = pd.concat(pl.starmap(sf.evaluate, zip(op, sz)), ignore_index=True)
    toc = time.time()

    took = toc - tic

    return took, sz, prf

tic = time.time()
with Pool(num) as pl:
    args  = zip(num * [pdk], num * [ckt], num * [net])
    syms  = pl.starmap(sf.operational_amplifier, args)
toc = time.time()
print(f'Creating took {toc - tic}s')

for i in range(10):
    times, sizes, perfs = test_perf_parallel(syms, num)
    sizes_df = pd.concat([sizes_df, pd.concat(sizes)])
    perfs_df = pd.concat([perfs_df, perfs])

    print(f'{i}: Creating {num} simulations took {times}s')

sizes_df = sizes_df.reset_index(drop=True)
perfs_df = perfs_df.reset_index(drop=True)

print(sizes_df)
print(perfs_df)


 
