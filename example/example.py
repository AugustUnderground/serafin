import serafin as sf

# MAKRE SURE `include: path` is updated in `example/gpdk180.yml`

pdk = './example/gpdk180.yml'
net = './example/sym.scs'
ckt = './example/sym.yml'
num = 10

## Single
sym = sf.make_op_amp(pdk, ckt, net)

prf = sf.evaluate(sym)

prf['performance']

## Parallel
syms = sf.make_op_amp(pdk, ckt, net, num)

prfs = sf.evaluate(syms)

prfs['performance']
