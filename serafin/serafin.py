""" Single-Ended Operational Amplifier Characterization """

from   functools        import reduce, partial
from   operator         import or_
from   collections.abc  import Iterable
from   shutil           import rmtree
from   typing           import NamedTuple, Union
import warnings
import errno
import yaml
from   scipy            import interpolate
import numpy            as np
import pandas           as pd
import pyspectre        as ps
from   pyspectre.core   import Session

from .util              import *

class OperationalAmplifier(NamedTuple):
    """
    Single Ended Operational Amplifier
    """
    session:     Iterable[Session]
    sim_dir:     Iterable[str]
    parameters:  dict[str, float]
    geometrical: dict[str, float]
    electrical:  dict[str, float]
    area_expr:   str
    constraints: dict[str, dict[str,float]]
    devices:     dict[str, str]
    op_params:   dict[str, str]
    of_params:   dict[str, str]

    def __del__(self):
        ps.stop_session(self.session, remove_raw = True)
        _ = list(map(rmtree, self.sim_dir))

def initial_sizing(op: OperationalAmplifier) -> pd.DataFrame:
    return pd.DataFrame.from_dict({k: [v] for k,v in op.geometrical.items()})

def random_sizing(op: OperationalAmplifier) -> pd.DataFrame:
    keys   = list(op.geometrical.keys())
    l_ids  = [k for k in keys if k.startswith('L')]
    w_ids  = [k for k in keys if k.startswith('W')]
    m_ids  = [k for k in keys if k.startswith('M')]
    s_num  = 1 if isinstance(op.session, Session) else len(op.session)
    l_num  = len(l_ids)
    w_num  = len(w_ids)
    m_num  = len(m_ids)

    l_min  = op.constraints['length']['min']
    l_max  = op.constraints['length']['max']
    l_grid = op.constraints['length']['grid']
    ls     = np.random.choice(np.arange(l_min, l_max, l_grid), (s_num, l_num))

    w_min  = op.constraints['width']['min']
    w_max  = op.constraints['width']['max']
    w_grid = op.constraints['width']['grid']
    ws     = np.random.choice(np.arange(w_min, w_max, w_grid), (s_num, w_num))

    m_min  = 1
    m_max  = 20
    m_grid = 1
    ms     = np.random.choice(np.arange(m_min, m_max, m_grid), (s_num, m_num))

    sizing = pd.DataFrame( np.hstack((ls, ws, ms))
                         , columns = (l_ids + w_ids + m_ids))

    return sizing

def make_op_amp( pdk_cfg: str, ckt_cfg: str, net: str, num: int = 1
               ) -> OperationalAmplifier:

    with open(ckt_cfg, mode = 'r', encoding = 'utf-8') as ckt_h:
        ckt = yaml.load(ckt_h, Loader=yaml.FullLoader)

    with open(pdk_cfg, mode = 'r', encoding = 'utf-8') as pdk_h:
        pdk = yaml.load(pdk_h, Loader=yaml.FullLoader)

    sim_dir     = [ setup_dir(pdk, net, ckt) for _ in range(num)
                  ] if num > 1 else setup_dir(pdk, net, ckt)
    net_path    = [ f'{sd}/tb.scs' for sd in sim_dir
                  ] if num > 1 else f'{sim_dir}/tb.scs'
    raw_path    = [ f'{rp}/op.raw' for rp in sim_dir
                  ] if num > 1 else f'{sim_dir}/op.raw'

    session     = ps.start_session(net_path, raw_path = raw_path)

    parameters  = pdk['testbench'] \
                | pdk['defaults']  \
                | ckt['parameters']['testbench']
    geometrical = ckt['parameters']['geometrical']
    electrical  = ckt['parameters']['electrical']
    area_expr   = ckt['parameters']['area']

    constraints = pdk['constraints']

    devices     = { d['id']: d['type'] for d in ckt['devices'] }
    dc_opps     = pdk['devices']['dcop']['parameters']
    op_pre      = pdk['devices']['dcop']['prefix']
    op_suf      = pdk['devices']['dcop']['suffix']

    op_params   = dict( reduce( or_
                              , [ { f'{op_pre}{d}{op_suf}:{op}': f'{d}_{op}'
                                    for op in dc_opps }
                                  for d in devices.keys() ] ) )

    of_devs     = pdk['devices']['dcmatch']
    of_params   = dict( reduce( or_
                              , [ { f'{of["prefix"]}{d}{of["suffix"]}' : f'{d}_{of["reference"]}'
                                    for of in of_devs[t] }
                                  for d,t in devices.items() ] ) )

    op_amp      = OperationalAmplifier( session     = session
                                      , sim_dir     = sim_dir
                                      , parameters  = parameters
                                      , geometrical = geometrical
                                      , electrical  = electrical
                                      , area_expr   = area_expr
                                      , constraints = constraints
                                      , devices     = devices
                                      , op_params   = op_params
                                      , of_params   = of_params
                                      , )
    return op_amp

def _current_sizing( op: OperationalAmplifier
                  ) -> Union[dict[str,float], Iterable[dict[str,float]]]:
    num    = 1 if isinstance(op.session, Session) else len(op.session)
    params = num * [op.geometrical.keys()] if num > 1 else op.geometrical.keys()
    return ps.get_parameters(op.session, params)

def current_sizing(op: OperationalAmplifier) -> pd.DataFrame:
    keys = list(op.geometrical.columns())
    vals = np.array([[s[k] for k in keys] for s in _current_sizing(op)])
    return pd.DataFrame(vals, columns = keys)

def _estimated_area( op: OperationalAmplifier
                   ) -> pd.DataFrame:
    num    = 1 if isinstance(op.session, Session) else len(op.session)
    params = _current_sizing(op) if num > 1 else [_current_sizing(op)]
    area   = np.array([eval(op.area_expr, {}, p) for p in params]).T
    return pd.DataFrame(area, columns = ['area'])

def _offset( op: OperationalAmplifier, dcmatch: pd.DataFrame
           ) -> pd.DataFrame:
    if dcmatch is None:
        df       = nan_frame(list(op.of_params.values()))
    else:
        perf_ids = { 'totalOutput.sigmaOut': 'voff_stat'
                   , 'totalOutput.dcOp':     'voff_syst' }
        perf     = dcmatch[list(perf_ids.keys())].rename(columns = perf_ids)
        offset   = dcmatch[list(op.of_params.keys())].rename(columns = op.of_params)
        df       = pd.concat([perf, offset], axis=1)
    return df

def _stability(stb: pd.DataFrame) -> pd.DataFrame:
    cols          = ['a_0', 'ugbw', 'pm', 'gm', 'cof']
    if stb is None:
        df        = nan_frame(cols)
    else:
        loop_gain = stb['loopGain'].values
        freq      = stb['freq'].values
        gain      = db20(loop_gain)
        phase     = np.angle(loop_gain, deg = True)
        a0db      = gain[0].item()
        a3db      = a0db - 3.0
        a0_idx    = find_closest_idx(gain, 0.0)
        f0db      = freq[a0_idx].real.item()
        f0_idx    = find_closest_idx(freq, f0db)
        ph0_idx   = find_closest_idx(phase, 0.0)
        pm        = phase[f0_idx]
        cof       = freq[ph0_idx].real.item()
        gm        = gain[ph0_idx]
        df        = pd.DataFrame( np.array([[a0db, f0db, pm, gm, cof]])
                                , columns = cols )
    return df

def _transient( tran: pd.DataFrame, vs: float = 0.5 ) -> dict[str, float]:
    cols           = ['sr_r', 'sr_f', 'os_r', 'os_f']
    if tran is None:
        df         = nan_frame(cols)
    else:
        time       = tran['time'].values
        out        = tran['OUT'].values

        idx_100    = find_closest_idx(time, 100e-9)
        idx_090    = find_closest_idx(time, 90e-6)
        idx_050    = find_closest_idx(time, 50e-6)
        idx_099    = find_closest_idx(time, 99.9e-6)

        rising     = out[idx_100:idx_050]
        falling    = out[(idx_050+1):idx_099]

        lower      = (0.1 * vs) - (vs / 2.0)
        upper      = (0.9 * vs) - (vs / 2.0)

        p1_rising  = time[find_closest_idx(rising, lower)]
        p2_rising  = time[find_closest_idx(rising, upper)]
        d_rising   = p2_rising - p1_rising

        sr_rising  = (upper - lower) / d_rising if d_rising > 0 else 0.0

        p1_falling = time[find_closest_idx(falling, upper)]
        p2_falling = time[find_closest_idx(falling, lower)]
        d_falling  = p2_falling - p1_falling

        sr_falling = (lower - upper) / d_falling if d_falling > 0 else 0.0

        os_rising  = 100 * (np.max(rising) - out[idx_050]) / (out[idx_050] - out[idx_100])
        os_falling = 100 * (np.min(falling) - out[idx_090]) / (out[idx_090] - out[idx_050])

        df         =  pd.DataFrame( np.array([[ sr_rising, sr_falling
                                              , os_rising, os_falling]])
                                  , columns = cols )
    return df

def _output_referred_noise(noise: pd.DataFrame) -> pd.DataFrame:
    cols     = ['vn_1Hz', 'vn_10Hz', 'vn_100Hz', 'vn_1kHz', 'vn_10kHz', 'vn_100kHz']
    if noise is None:
        df   = nan_frame(cols)
    else:
        fs   = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
        freq = noise['freq'].values
        out  = noise['out'].values
        vn   = interpolate.pchip_interpolate(freq, out, fs).reshape(1,-1)
        df   =  pd.DataFrame(vn, columns = cols)
    return df

def _out_swing_dc( dc1: pd.DataFrame, vdd: float = 1.8, dev: float = 1.0e-4
                 ) -> pd.DataFrame:
    cols          = ['v_oh', 'v_ol']
    if dc1 is None:
        df        = nan_frame(cols)
    else:
        out       = dc1['OUT'].values
        out_ideal = dc1['OUT_IDEAL'].values
        vid       = dc1['vid'].values
        out_dc    = (out - out[find_closest_idx(vid, 0.0)])
        dev_rel   = np.abs(out_dc - out_ideal) / vdd
        vil_idx   = np.argmin(np.abs(dev_rel[vid <= 0.0] - dev))
        vih_idx   = np.argmin(np.abs(dev_rel[vid >= 0.0] - dev))
        vol_dc    = (out_dc[vil_idx] + (vdd / 2.0)).item()
        voh_dc    = (out_dc[vih_idx] + (vdd / 2.0)).item()
        df        =  pd.DataFrame(np.array([[voh_dc, vol_dc]]), columns = cols)
    return df

def _rejection(xf: pd.DataFrame) -> pd.DataFrame:
    cols         = ['psrr_p', 'psrr_n', 'cmrr']
    if xf is None:
        df       = nan_frame(cols)
    else:
        vid_db   = db20(xf['VID'].values)
        vicm_db  = db20(xf['VICM'].values)
        vsupp_db = db20(xf['VSUPP'].values)
        vsupn_db = db20(xf['VSUPN'].values)
        psrr_p   = (vid_db - vsupp_db)[0].item()
        psrr_n   = (vid_db - vsupn_db)[0].item()
        cmrr     = (vid_db - vicm_db)[0].item()
        df       = pd.DataFrame( np.array([[psrr_p, psrr_n, cmrr]])
                               , columns = cols )
    return df

def _out_swing_ac( ac: pd.DataFrame, A3dB: float, vdd: float = 1.8
                 ) -> pd.DataFrame:
    cols       = ['v_ih', 'v_il']
    if ac is None:
        df     = nan_frame(cols)
    else:
        vicm   = ac['vicm'].values
        out_ac = db20(ac['OUT'].values)
        vil_ac = (vicm[np.argmin(np.abs(out_ac[vicm <= 0.0] - A3dB))] + (vdd / 2.0)
                 ).real.item()
        vih_ac = (vicm[np.argmin(np.abs(out_ac[vicm >= 0.0] - A3dB))] + (vdd / 2.0)
                 ).real.item()
        df     = pd.DataFrame(np.array([[vih_ac, vil_ac ]]), columns = cols)
    return df

def _output_current(dc3: pd.DataFrame, dc4: pd.DataFrame) -> pd.DataFrame:
    cols          = ['i_out_min', 'i_out_max']
    if (dc3 is None) or (dc4 is None):
        df        = nan_frame(cols)
    else:
        i_out_min = dc3['DUT:O'].values[0].item()
        i_out_max = dc4['DUT:O'].values[0].item()
        df        =  pd.DataFrame( np.array([[i_out_min, i_out_max]])
                                 , columns = cols )
    return df

def _operating_point( op: OperationalAmplifier, dcop: pd.DataFrame
                    ) -> pd.DataFrame:
    if dcop is None:
        df       = nan_frame(list(op.params.values()) + ['idd', 'iss'])
    else:
        perf_ids = {'DUT:VDD': 'idd', 'DUT:VSS': 'iss'}
        perf     = dcop[list(perf_ids.keys())].rename(columns = perf_ids)
        ops      = dcop[list(op.op_params.keys())].rename(columns = op.op_params)
        df       = pd.concat([perf, ops], axis = 1)
    return df

def evaluate( op: OperationalAmplifier, sizing: pd.DataFrame = None
            ) -> pd.DataFrame:
    num     = 1 if isinstance(op.session, Session) else len(op.session)

    s_dict  = sizing.to_dict(orient = 'list') if sizing is not None else {}
    sizes   = [dict(zip(s_dict.keys(), col)) for col in zip(*s_dict.values())]
    ret     = num * [True] if sizing is None else \
                ps.set_parameters(op.session, sizes if num > 1 else sizes[0])

    if not all(ret):
        raise(IOError( errno.EIO, os.strerror(errno.EIO)
                     , f'spectre failed to set sizing parameters with non-zero exit code {ret}.' ))

    vdd     = op.parameters['vdd']
    dev     = op.parameters['dev']

    results = ps.run_all(op.session) if num > 1 else [ps.run_all(op.session)]

    if not all(map(bool, results)):
        warnings.warn('Some simulations failed', RuntimeWarning)

    area    = _estimated_area(op)
    dcmatch = pd.concat( [_offset(op, result.get('dcmatch', None)) for result in results]
                       , ignore_index = True )
    stb     = pd.concat( [_stability(result['stb']) for result in results]
                       , ignore_index = True )
    tran    = pd.concat( [ _transient(result['tran'], op.parameters['vs'])
                           for result in results ]
                       , ignore_index = True )
    noise   = pd.concat( [ _output_referred_noise(result['noise'])
                           for result in results ]
                       , ignore_index = True )
    dc1     = pd.concat( [ _out_swing_dc(result['dc1'], vdd, dev)
                           for result in results ]
                       , ignore_index = True )
    ac      = pd.concat( [ _out_swing_ac( result['ac']
                                        , (stb.iloc[[idx]][['a_0']] - 3.0
                                          ).values[0].item()
                                        , vdd )
                          for idx,result in enumerate(results) ]
                       , ignore_index = True )
    xf      = pd.concat( [ _rejection(result['xf']) for result in results ]
                       , ignore_index = True )
    dc34    = pd.concat( [ _output_current(result['dc3'], result['dc4'])
                           for result in results ]
                       , ignore_index = True )
    dcop    = pd.concat( [ _operating_point(op, result['dcop'])
                           for result in results ]
                       , ignore_index = True )

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
