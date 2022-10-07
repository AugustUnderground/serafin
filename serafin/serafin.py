""" Single-Ended Operational Amplifier Characterization """

from   functools       import reduce, partial
from   operator        import or_
from   tempfile        import mkdtemp
from   shutil          import copyfile, rmtree
from   collections.abc import Iterable
from   typing          import NamedTuple, Union
import os
import yaml
from   scipy           import interpolate
import numpy           as np
import pandas          as pd
import pyspectre       as ps
from   pyspectre.core  import Session

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
    devices:     dict[str, str]
    op_params:   dict[str, str]
    of_params:   dict[str, str]

    def __del__(self):
        ps.stop_session(self.session, remove_raw = True)
        _ = list(map(rmtree, self.sim_dir))

def _setup_dir(pdk_cfg: dict, net_scs: str, ckt_cfg: dict) -> str:
    """
    Setup a temporary simulation directory with testbench and subckt ready to go.
    """

    cwd    = os.path.dirname(os.path.abspath(__file__))
    op_id  = os.path.basename(net_scs).split('.')[0]
    usr_id = os.getlogin()

    with open(net_scs, mode = 'r', encoding = 'utf-8') as net_h:
        net = net_h.read()

    includes = '\n'.join( [ f'include "{p["path"]}" section={p["section"]}'
                            for p in pdk_cfg.get('include', {}) ] )

    defaults  =  ckt_cfg.get( 'parameters', {}).get('testbench', {}
                           ) | pdk_cfg.get('testbench', {})

    tb_params = 'parameters ' \
              + ' '.join([ f'{p}={v}' for p,v in defaults.items() ])

    op_params = 'parameters ' \
              + ' '.join([ f'{p}={v}'
                           for p,v in ckt_cfg.get( 'parameters', {}
                                                ).get( 'geometrical'
                                                     , {} ).items() ] )

    area      = ckt_cfg.get('parameters', {}).get('area', '0.0')
    ae_params = f'parameters area={area}' if area != '0.0' else ''

    op_pre    = pdk_cfg['devices']['dcop']['prefix']
    op_suf    = pdk_cfg['devices']['dcop']['suffix']
    op_par    = pdk_cfg['devices']['dcop']['parameters']
    saves     = 'save ' \
              + '\\\n\t'.join([ f'{op_pre}*{op_suf}:{param}'
                                for param in op_par ])

    subckt    = '\n\n'.join([ includes, tb_params, op_params, ae_params
                            , net, saves ])

    tmp_dir   = mkdtemp(prefix = f'{usr_id}_{op_id}_')

    with open(f'{tmp_dir}/op.scs', 'w',  encoding = 'utf-8') as sub_h:
        sub_h.write(subckt)

    copyfile(f'{cwd}/resource/testbench.scs', f'{tmp_dir}/tb.scs')

    return tmp_dir

def make_op_amp( pdk_cfg: str, ckt_cfg: str, net: str, num: int = 1
               ) -> OperationalAmplifier:

    with open(ckt_cfg, mode = 'r', encoding = 'utf-8') as ckt_h:
        ckt = yaml.load(ckt_h, Loader=yaml.FullLoader)

    with open(pdk_cfg, mode = 'r', encoding = 'utf-8') as pdk_h:
        pdk = yaml.load(pdk_h, Loader=yaml.FullLoader)

    sim_dir    = [_setup_dir(pdk, net, ckt) for _ in range(num)] \
                    if num > 1 else _setup_dir(pdk, net, ckt)
    net_path   = [ f'{sd}/tb.scs' for sd in sim_dir ] \
                   if num > 1 else f'{sim_dir}/tb.scs'
    raw_path   = [ f'{rp}/op.raw' for rp in sim_dir ] \
                   if num > 1 else f'{sim_dir}/op.raw'

    session    = ps.start_session(net_path, raw_path = raw_path)

    parameters  = pdk['testbench']  \
                | pdk['defaults']   \
                | ckt['parameters']['testbench']
    geometrical = ckt['parameters']['geometrical']
    electrical  = ckt['parameters']['electrical']
    area_expr   = ckt['parameters']['area']
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
                                      , devices     = devices
                                      , op_params   = op_params
                                      , of_params   = of_params
                                      , )
    return op_amp

def current_sizing( op: OperationalAmplifier
                  ) -> Union[dict[str,float], Iterable[dict[str,float]]]:
    num    = 1 if isinstance(op.session, Session) else len(op.session)
    params = num * [op.geometrical.keys()] if num > 1 else op.geometrical.keys()
    return ps.get_parameters(op.session, params)

def _find_closest_idx(array: np.array, value: float) -> int:
    return np.argmin(np.abs(array - value))

def _db20(x: Union[float, np.array]) -> Union[float, np.array]:
    return np.log10(np.abs(x)) * 20.0

def _estimated_area( op: OperationalAmplifier
                   ) -> Iterable[dict[str, float]]:
    num  = 1 if isinstance(op.session, Session) else len(op.session)
    params = current_sizing(op) if num > 1 else [current_sizing(op)]
    area = [{ 'area': eval(op.area_expr, {}, p) } for p in params]
    return area

def _offset( op: OperationalAmplifier, dcmatch: pd.DataFrame
           ) -> dict[str, float]:
    voff_stat = dcmatch['totalOutput.sigmaOut'].values[0].item()
    voff_syst = dcmatch['totalOutput.dcOp'].values[0].item()
    offset    = { op.of_params[of]: dcmatch[of].values.item()
                  for of in dcmatch.columns if of in op.of_params.keys() }
    return { 'perf' : { 'voff_stat': voff_stat
                      , 'voff_syst': voff_syst }
           , 'offset': offset }

def _stability(stb: pd.DataFrame) -> dict[str, float]:
    loop_gain = stb['loopGain'].values
    freq      = stb['freq'].values
    gain      = _db20(loop_gain)
    phase     = np.angle(loop_gain, deg = True)
    a0db      = gain[0].item()
    a3db      = a0db - 3.0
    a0_idx    = _find_closest_idx(gain, 0.0)
    f0db      = freq[a0_idx].real.item()
    f0_idx    = _find_closest_idx(freq, f0db)
    ph0_idx   = _find_closest_idx(phase, 0.0)
    pm        = phase[f0_idx]
    cof       = freq[ph0_idx].real.item()
    gm        = gain[ph0_idx]
    return { 'a_0': a0db
           , 'ugbw': f0db
           , 'pm': pm
           , 'gm': gm
           , 'cof': cof }

def _transient( tran: pd.DataFrame, vs: float = 0.5 ) -> dict[str, float]:
    time          = tran['time'].values
    out           = tran['OUT'].values

    idx_100       = _find_closest_idx(time, 100e-9)
    idx_090       = _find_closest_idx(time, 90e-6)
    idx_050       = _find_closest_idx(time, 50e-6)
    idx_099       = _find_closest_idx(time, 99.9e-6)

    rising        = out[idx_100:idx_050]
    falling       = out[(idx_050+1):idx_099]

    lower         = (0.1 * vs) - (vs / 2.0)
    upper         = (0.9 * vs) - (vs / 2.0)

    p1_rising     = time[_find_closest_idx(rising, lower)]
    p2_rising     = time[_find_closest_idx(rising, upper)]
    d_rising      = p2_rising - p1_rising

    sr_rising     = (upper - lower) / d_rising if d_rising > 0 else 0.0

    p1_falling    = time[_find_closest_idx(falling, upper)]
    p2_falling    = time[_find_closest_idx(falling, lower)]
    d_falling     = p2_falling - p1_falling

    sr_falling    = (lower - upper) / d_falling if d_falling > 0 else 0.0

    os_rising     = 100 * (np.max(rising) - out[idx_050]) / (out[idx_050] - out[idx_100])
    os_falling    = 100 * (np.min(falling) - out[idx_090]) / (out[idx_090] - out[idx_050])

    return { 'sr_r': sr_rising
           , 'sr_f': sr_falling
           , 'os_r': os_rising
           , 'os_f': os_falling }

def _output_referred_noise( noise: pd.DataFrame ) -> dict[str, float]:
    ids  = ['vn_1Hz', 'vn_10Hz', 'vn_100Hz', 'vn_1kHz', 'vn_10kHz', 'vn_100kHz']
    fs   = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
    freq = noise['freq'].values
    out  = noise['out'].values
    vn   = interpolate.pchip_interpolate(freq, out, fs)
    return dict(zip(ids,vn))

def _out_swing_dc( dc1: pd.DataFrame, vdd: float = 1.8, dev: float = 1.0e-4
                 ) -> dict[str, float]:
    out       = dc1['OUT'].values
    out_ideal = dc1['OUT_IDEAL'].values
    vid       = dc1['vid'].values
    out_dc    = (out - out[_find_closest_idx(vid, 0.0)])
    dev_rel   = np.abs(out_dc - out_ideal) / vdd
    vil_idx   = np.argmin(np.abs(dev_rel[vid <= 0.0] - dev))
    vih_idx   = np.argmin(np.abs(dev_rel[vid >= 0.0] - dev))
    vol_dc    = (out_dc[vil_idx] + (vdd / 2.0))
    voh_dc    = (out_dc[vih_idx] + (vdd / 2.0))
    return { 'v_oh': voh_dc
           , 'v_ol': vol_dc }

def _rejection( xf: pd.DataFrame ) -> dict[str, float]:
    vid_db   = _db20(xf['VID'].values)
    vicm_db  = _db20(xf['VICM'].values)
    vsupp_db = _db20(xf['VSUPP'].values)
    vsupn_db = _db20(xf['VSUPN'].values)
    psrr_p   = (vid_db - vsupp_db)[0]
    psrr_n   = (vid_db - vsupn_db)[0]
    cmrr     = (vid_db - vicm_db)[0]
    return { 'psrr_p': psrr_p
           , 'psrr_n': psrr_n
           , 'cmrr': cmrr }

def _out_swing_ac( ac: pd.DataFrame, A3dB: float, vdd: float = 1.8
                 ) -> dict[str, float]:
    vicm   = ac['vicm'].values
    out_ac = _db20(ac['OUT'].values)
    vil_ac = (vicm[np.argmin(np.abs(out_ac[vicm <= 0.0] - A3dB))] + (vdd / 2.0)).real
    vih_ac = (vicm[np.argmin(np.abs(out_ac[vicm >= 0.0] - A3dB))] + (vdd / 2.0)).real
    return { 'v_ih': vih_ac
           , 'v_il': vil_ac }

def _output_current( dc3: pd.DataFrame, dc4: pd.DataFrame
                   ) -> dict[str, float]:
    i_out_min = dc3['DUT:O'].values[0]
    i_out_max = dc4['DUT:O'].values[0]
    return { 'i_out_min': i_out_min
           , 'i_out_max': i_out_max }

def _operating_point( op: OperationalAmplifier
                    , dcop: pd.DataFrame
                    ) -> dict[str, float]:
    idd = dcop['DUT:VDD'].values.item()
    iss = dcop['DUT:VSS'].values.item()
    ops = { op.op_params[o]: dcop[o].values.item()
            for o in dcop.columns
            if o in op.op_params.keys() }
    return { 'perf': { 'idd': idd
                     , 'iss': iss }
           , 'op': ops }

def evaluate(op: OperationalAmplifier) -> dict[str, float]:
    num     = 1 if isinstance(op.session, Session) else len(op.session)
    vdd     = op.parameters['vdd']
    dev     = op.parameters['dev']

    results = ps.run_all(op.session) if num > 1 else [ps.run_all(op.session)]
    area    = _estimated_area(op)
    dcmatch = [ _offset(op, result['dcmatch']) for result in results ]
    stb     = [ _stability(result['stb']) for result in results ]
    tran    = [ _transient(result['tran'], op.parameters['vs'])
                for result in results ]
    noise   = [ _output_referred_noise(result['noise']) for result in results ]
    dc1     = [ _out_swing_dc(result['dc1'], vdd, dev) for result in results ]
    ac      = [ _out_swing_ac(result['ac'], (stab['a_0'] - 3.0), vdd)
                for result,stab in zip(results,stb) ]
    xf      = [ _rejection(result['xf']) for result in results ]
    dc34    = [ _output_current(result['dc3'], result['dc4'])
                for result in results ]
    dcop    = [ _operating_point(op, result['dcop']) for result in results ]

    perf    = list( map( partial(reduce, or_)
                       , zip( area
                            , [ dcm['perf'] for dcm in dcmatch ]
                            , stb
                            , tran
                            , noise
                            , dc1
                            , ac
                            , xf
                            , dc34
                            , [ dco['perf'] for dco in dcop ] ) ) )

    return { 'performance': perf if num > 1 else perf[0]
           , 'operating_point': [ dco['op'] for dco in dcop ] \
                                if num > 1 else dcop[0]['op']
           , 'offset': [ dcm['offset'] for dcm in dcmatch ] \
                       if num > 1 else dcmatch[0]['offset'] }
