""" Single-Ended Operational Amplifier Characterization """

from   functools        import reduce
from   operator         import or_
from   shutil           import rmtree
from   typing           import NamedTuple
import warnings
import errno
import yaml
import numpy            as np
import pandas           as pd
import pyspectre        as ps

from .util              import *

class OperationalAmplifier(NamedTuple):
    """
    Single Ended Operational Amplifier
    """
    session:      ps.Session
    sim_dir:      str
    parameters:   dict[str, float]
    geom_init:    dict[str, float]
    area_expr:    str
    constraints:  dict[str, dict[str,float]]
    devices:      dict[str, str]
    dcop_params:  dict[str, str]
    offs_params:  dict[str, str]
    performances: dict[str, str]

    def __del__(self):
        ps.stop_session(self.session, remove_raw = True)
        rmtree(self.sim_dir)

def operational_amplifier( pdk_cfg: str, ckt_cfg: str, net: str
                         ) -> OperationalAmplifier:

    with open(ckt_cfg, mode = 'r', encoding = 'utf-8') as ckt_h:
        ckt = yaml.load(ckt_h, Loader=yaml.FullLoader)

    with open(pdk_cfg, mode = 'r', encoding = 'utf-8') as pdk_h:
        pdk = yaml.load(pdk_h, Loader=yaml.FullLoader)

    sim_dir     = setup_dir(pdk, net, ckt)
    net_path    = f'{sim_dir}/tb.scs'
    raw_path    = f'{sim_dir}/op.raw'

    session     = ps.start_session(net_path, raw_path = raw_path)

    parameters  = pdk['defaults'] \
                | pdk['testbench'] \
                | ckt['parameters']['testbench']
    geom_init   = ckt['parameters']['geometrical']
    area_expr   = ckt['parameters']['area']

    constraints = pdk['constraints']

    devices     = { d['id']: d['type'] for d in ckt['devices'] }
    dc_opps     = pdk['devices']['dcop']['parameters']
    op_pre      = pdk['devices']['dcop']['prefix']
    op_suf      = pdk['devices']['dcop']['suffix']

    dcop_params = dict( reduce( or_
                              , [ { f'{op_pre}{d}{op_suf}:{op}': f'{d}_{op}'
                                    for op in dc_opps }
                                  for d,t in devices.items()
                                  if t not in  ["CAP", "RES"] ] ) )

    of_devs     = pdk['devices']['dcmatch']
    offs_params = dict( reduce( or_
                              , [ { f'{of["prefix"]}{d}{of["suffix"]}' : f'{d}_{of["reference"]}'
                                    for of in of_devs[t] }
                                  for d,t in devices.items() 
                                  if t not in  ["CAP", "RES"] ] ) )

    op_amp      = OperationalAmplifier( session      = session
                                      , sim_dir      = sim_dir
                                      , parameters   = parameters
                                      , geom_init    = geom_init
                                      , area_expr    = area_expr
                                      , constraints  = constraints
                                      , devices      = devices
                                      , dcop_params  = dcop_params
                                      , offs_params  = offs_params
                                      , performances = PERFORMANCE_PARAMETERS
                                      , )
    ps.set_parameters(session, parameters | geom_init)

    return op_amp

def initial_sizing(op: OperationalAmplifier) -> pd.DataFrame:
    return from_dict(op.geom_init)

def random_sizing(op: OperationalAmplifier) -> pd.DataFrame:
    keys   = list(op.geom_init.keys())
    l_ids  = [k for k in keys if k.startswith('L')]
    w_ids  = [k for k in keys if k.startswith('W')]
    m_ids  = [k for k in keys if k.startswith('M')]
    l_num  = len(l_ids)
    w_num  = len(w_ids)
    m_num  = len(m_ids)

    l_min  = op.constraints['length']['min']
    l_max  = op.constraints['length']['max']
    l_grid = op.constraints['length']['grid']
    ls     = np.random.choice(np.arange(l_min, l_max, l_grid), (1, l_num))

    w_min  = op.constraints['width']['min']
    w_max  = op.constraints['width']['max']
    w_grid = op.constraints['width']['grid']
    ws     = np.random.choice(np.arange(w_min, w_max, w_grid), (1, w_num))

    m_min  = 1
    m_max  = 20
    m_grid = 1
    ms     = np.random.choice(np.arange(m_min, m_max, m_grid), (1, m_num))

    sizing = pd.DataFrame( np.hstack((ls, ws, ms))
                         , columns = (l_ids + w_ids + m_ids))

    return sizing

def _current_sizing(op: OperationalAmplifier) -> dict[str,float]:
    """
    Get current sizing parameters from netlist in active session as dict
    """
    return ps.get_parameters(op.session, list(op.geom_init.keys()))

def current_sizing(op: OperationalAmplifier) -> pd.DataFrame:
    """
    Get current sizing parameters from netlist in active session
    """
    return from_dict(_current_sizing(op))

def _set_paramters( op: OperationalAmplifier, sizing: pd.DataFrame
                  , tol: float = 1e-10 ) -> bool:
    cols  = sorted(list(sizing.columns))
    lmin  = op.constraints['length']['min']
    lmax  = op.constraints['length']['max']
    wmin  = op.constraints['width']['min']
    wmax  = op.constraints['width']['max']
    mmin  = 1
    mmax  = 42
    mins  = np.array([ lmin if c.startswith('L') else
                       wmin if c.startswith('W') else
                       mmin for c in cols ])
    maxs  = np.array([ lmax if c.startswith('L') else
                       wmax if c.startswith('W') else
                       mmax for c in cols ])
    vals  = np.clip(sizing[cols].values[0], mins, maxs)[None,:]
    sizes = pd.DataFrame(vals, columns = cols)
    ret   = ps.set_parameters(op.session, to_dict(sizes))

    if not ret:
        msg = f'spectre failed to set sizing parameters with non-zero exit code {ret}.'
        raise(IOError(errno.EIO, os.strerror(errno.EIO), msg))

    return np.all(np.abs(sizes[cols].values - sizing[cols].values) < tol)

def evaluate( op: OperationalAmplifier, sizing: pd.DataFrame = None
            , dcop_only: bool = False ) -> pd.DataFrame:
    """
    Evaluate the operational amplifier with a given sizing. Optionally only
    determines the operating point, which is faster.
    """

    if sizing is not None:
        ret = _set_paramters(op, sizing)
        if not ret:
            warnings.warn('Clipped sizing', RuntimeWarning)

    if dcop_only:
        results = ps.run_analysis(op.session, 'dcop')
    else:
        results = ps.run_all(op.session)

    if not bool(results):
        msg = 'Simulations Failed.'
        raise(IOError(errno.EIO, os.strerror(errno.EIO), msg))

    if dcop_only:
        perf = operating_point(results['dcop'], op.dcop_params)
    else:
        perf = extract_performance(op, results)

    return perf

def extract_performance( op: OperationalAmplifier
                       , results: dict[str, pd.DataFrame]
                       ) -> pd.DataFrame:
    """
    Extract performance from simulation results
    """
    vdd     = op.parameters['vdd']
    dev     = op.parameters['dev']
    dcmatch = offset(results['dcmatch'], op.offs_params)
    stb     = stability(results['stb'])
    tran    = transient(results['tran'])
    noise   = output_referred_noise(results['noise'])
    dc1     = out_swing_dc(results['dc1'], vdd, dev)
    ac      = out_swing_ac(results['ac'], stb['a_0'].values.item() - 3.0, vdd)
    xf      = rejection(results['xf'])
    dc34    = output_current(results['dc3'], results['dc4'])
    dcop    = operating_point(results['dcop'], op.dcop_params)
    area    = estimated_area(op.area_expr, current_sizing(op))
    perf    = pd.concat( [ area
                         , dcmatch
                         , stb
                         , tran
                         , noise
                         , dc1
                         , ac
                         , xf
                         , dc34
                         , dcop ]
                       , axis = 1 )
    return perf
