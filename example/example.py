import serafin as sf

pdk = './example/gpdk180.yml'
net = './example/sym.scs'
ckt = './example/sym.yml'

sym = sf.make_op_amp(pdk, ckt, net)

prf = sf.evaluate(sym)

prf['performance']
