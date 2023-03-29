# SERAFIN

**S**ingle-**E**nded Ope**r**ational **A**mpli**f**ier Character**i**zatio**n**
loosely based on [AC²E](https://github.com/electronics-and-drives/ace).

## Dependencies:

- [pynut](https://github.com/augustunderground/pynut)
- [pyspectre](https://github.com/augustunderground/pyspectre)

## Installation

With `pip`:

```sh
$ pip install git+https://github.com/augustunderground/serafin.git
```

From source:

```sh
$ git clone https://github.com/augustunderground/serafin.git
$ pushd serafin
$ pip install .
```

## Example

A few things need to be adjusted to run the example:

- Correct PDK path in `./example/gpdk180.yml`
- Derived statements for transistors in `./example/sym.scs`
- Change `nmos` and `pmos` name in `./example/sym.scs` according to PDK

```python
import serafin as sf

pdk = './example/gpdk180.yml'
net = './example/sym.scs'
ckt = './example/sym.yml'

sym = sf.operational_amplifier(pdk, ckt, net)

rsz = sf.random_sizing(sym)
prf = sf.evaluate(sym, rsz)
```
