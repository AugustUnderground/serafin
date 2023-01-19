""" Single-Ended Operational Amplifier Characterization Utility functions """

import os
from   tempfile        import mkdtemp
from   shutil          import copyfile
from   typing          import Union, Iterable
from   scipy           import interpolate
import numpy           as np
import pandas          as pd

def setup_dir(pdk_cfg: dict, net_scs: str, ckt_cfg: dict) -> str:
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

def repeat_values(d: dict[str, float], n: int) -> dict[str, list[float]]:
    return { k: n * [v] for k,v in d }

def transpose_dict(ds: Iterable[dict[str, float]]) -> dict[str, Iterable[float]]:
    keys = {d.keys() for d in ds}
    vals = np.array([list(d.values()) for d in ds]).T.tolist()
    return dict(zip(keys, vals))

def find_closest_idx(array: np.array, value: float) -> int:
    return np.argmin(np.abs(array - value), axis = len(array.shape) - 1)

def find_first_idx(array: np.array, value: float, edge: str) -> Union[int, None]:
    if edge == 'r':
        indices = np.argwhere(array > value)
    elif edge == 'f':
        indices = np.argwhere(array < value)
    else:
        indices = np.empty(0)
    return np.min(indices).item() if np.size(indices) else None

def db20(x: Union[float, np.array]) -> Union[float, np.array]:
    return np.log10(np.abs(x)) * 20.0

def nan_frame(cols: Iterable[str]) -> pd.DataFrame:
    length = len(cols)
    return pd.DataFrame(np.full((1,length), np.NaN), columns = cols)

def from_dict(d: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame.from_dict({k: [v] for k,v in d.items()})

def to_dict(df: pd.DataFrame) -> dict[str, float]:
    return {k: v[0] for k,v in df.to_dict(orient = 'list').items()}

def find_sr_r(times: np.array, values: np.array, upper: float, lower: float) -> np.float:

    rising_hi     = find_first_idx(values, upper, 'r')
    rising_lo     = find_first_idx(values, lower, 'r')

    if rising_hi and rising_lo:
        d_rising  = times[rising_hi]-times[rising_lo]
        sr_r      = (upper - lower) / d_rising if d_rising > 0 else np.nan
    else:
        sr_r      = np.nan
    return sr_r

def find_sr_f(times: np.array, values: np.array, upper: float, lower: float) -> np.float:
    flipped       = np.flip(values)
    candidate     = find_first_idx(flipped, upper, 'r')
    falling_hi    = (-1* candidate) -1 if candidate else None

    window        = values[falling_hi:]
    candidate     = find_first_idx(window, lower, 'f')
    falling_lo    = (falling_hi + candidate) if (candidate and falling_hi) else None

    if falling_hi and falling_lo:
        d_falling = times[falling_lo]-times[falling_hi]
        sr_f      = (lower - upper) / d_falling if d_falling > 0 else np.nan
    else:
        sr_f      = np.nan
    return sr_f

PERFORMANCE_PARAMETERS: dict[str,str] = { 'area':       'Estimated Area'
                                        , 'a_0':        'DC Loop Gain'
                                        , 'cmrr':       'Common Mode Rejection Ratio'
                                        , 'cof':        'Cross-Over Frequency'
                                        , 'gm':         'Gain Margin'
                                        , 'i_out_max':  'Maximum output Current'
                                        , 'i_out_min':  'Minimum output Current'
                                        , 'idd':        'Current Consumption'
                                        , 'iss':        'Current Consumption'
                                        , 'os_f':       'Overshoot Falling'
                                        , 'os_r':       'Overshoot Rising'
                                        , 'pm':         'Phase Margin'
                                        , 'psrr_n':     'Power Supply Rejection Ratio'
                                        , 'psrr_p':     'Power Supply Rejection Ratio'
                                        , 'sr_f':       'Slew Rate Falling'
                                        , 'sr_r':       'Slew Rate Rising'
                                        , 'ugbw':       'Unity Gain Bandwidth'
                                        , 'v_ih':       'Input Voltage Hight'
                                        , 'v_il':       'Input Voltage Low'
                                        , 'v_oh':       'Output Voltage High'
                                        , 'v_ol':       'Output Voltage Low'
                                        , 'vn_1Hz':     'Output Referred Noise @ 1Hz'
                                        , 'vn_10Hz':    'Output Referred Noise @ 10Hz'
                                        , 'vn_100Hz':   'Output Referred Noise @ 100Hz'
                                        , 'vn_1kHz':    'Output Referred Noise @ 1kHz'
                                        , 'vn_10kHz':   'Output Referred Noise @ 10kHz'
                                        , 'vn_100kHz':  'Output Referred Noise @ 100kHz'
                                        , 'voff_stat':  'Statistical Offset'
                                        , 'voff_syst':  'Systematic Offset'
                                        , }

def offset( dcmatch: pd.DataFrame, offset_params: dict[str, str]
          ) -> pd.DataFrame:
    col_ids  = { 'totalOutput.sigmaOut': 'voff_stat'
               , 'totalOutput.dcOp':     'voff_syst'
               } | offset_params

    left     = [ cn for co,cn in col_ids.items()
                    if co not in list(dcmatch.columns) ]
    right    = [ c for c in col_ids.keys() if c in list(dcmatch.columns) ]

    offset_  = dcmatch[right].rename(columns = col_ids)
    offset   = pd.concat( [ offset_
                          , pd.DataFrame( np.zeros((1,len(left)))
                                        , columns = left)]
                        , axis = 1
                        , ) if left else offset_
    return offset

def stability(stb: pd.DataFrame) -> pd.DataFrame:
    cols      = ['a_0', 'ugbw', 'pm', 'gm', 'cof']
    loop_gain = stb['loopGain'].values
    freq      = stb['freq'].values
    gain      = db20(loop_gain)
    phase     = np.angle(loop_gain, deg = True)
    a0_idx    = find_first_idx(gain, 0.0)
    ph0_idx   = find_first_idx(phase, 0.0)
    if a0_idx and ph0_idx:
        a0db      = gain[0].item()
        a3db      = a0db - 3.0
        f0db      = freq[a0_idx].real.item()
        f0_idx    = find_closest_idx(freq, f0db)
        pm        = phase[f0_idx].item()
        cof       = freq[ph0_idx].real.item()
        gm        = gain[ph0_idx].item()
    else:
        a0db      = np.nan
        f0db      = np.nan
        pm        = np.nan
        gm        = np.nan
        cof       = np.nan
    stability = pd.DataFrame( np.array([[a0db, f0db, pm, gm, cof]])
                            , columns = cols )
    return stability

def transient( tran: pd.DataFrame, vs: float = 0.5 ) -> dict[str, float]:
    cols       = ['sr_r', 'sr_f', 'os_r', 'os_f']
    time       = tran['time'].values
    out        = tran['OUT'].values

    idx_100    = find_closest_idx(time, 100e-9)
    idx_090    = find_closest_idx(time, 90e-6)
    idx_050    = find_closest_idx(time, 50e-6)
    idx_099    = find_closest_idx(time, 99.9e-6)

    rising     = out[idx_100:idx_050]
    falling    = out[(idx_050+1):idx_099]

    time_r     = time[idx_100:idx_050]
    time_f     = time[(idx_050+1):idx_099]

    lower      = (0.1 * vs) - (vs / 2.0)
    upper      = (0.9 * vs) - (vs / 2.0)

    sr_rising  = find_sr_r(time_r, rising, upper, lower)
    sr_falling = find_sr_f(time_f, falling, upper, lower)

    os_rising  = 100 * (np.max(rising) - out[idx_050]) / (out[idx_050] - out[idx_100])
    os_falling = 100 * (np.min(falling) - out[idx_090]) / (out[idx_090] - out[idx_050])

    transient  =  pd.DataFrame( np.array([[ sr_rising, sr_falling
                                          , os_rising, os_falling]])
                              , columns = cols )
    return transient

def output_referred_noise(noise: pd.DataFrame) -> pd.DataFrame:
    cols  = ['vn_1Hz', 'vn_10Hz', 'vn_100Hz', 'vn_1kHz', 'vn_10kHz', 'vn_100kHz']
    fs    = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
    freq  = noise['freq'].values
    out   = noise['out'].values
    vn    = interpolate.pchip_interpolate(freq, out, fs).reshape(1,-1)
    noise = pd.DataFrame(vn, columns = cols)
    return noise

def out_swing_dc( dc1: pd.DataFrame, vdd: float = 1.8, dev: float = 1.0e-4
                ) -> pd.DataFrame:
    cols      = ['v_oh', 'v_ol']
    out       = dc1['OUT'].values
    out_ideal = dc1['OUT_IDEAL'].values
    vid       = dc1['vid'].values
    out_dc    = (out - out[find_closest_idx(vid, 0.0)])
    dev_rel   = np.abs(out_dc - out_ideal) / vdd
    vil_idx   = np.argmin(np.abs(dev_rel[vid <= 0.0] - dev))
    vih_idx   = np.argmin(np.abs(dev_rel[vid >= 0.0] - dev))
    vol_dc    = (out_dc[vil_idx] + (vdd / 2.0)).item()
    voh_dc    = (out_dc[vih_idx] + (vdd / 2.0)).item()
    swing     =  pd.DataFrame(np.array([[voh_dc, vol_dc]]), columns = cols)
    return swing

def rejection(xf: pd.DataFrame) -> pd.DataFrame:
    cols     = ['psrr_p', 'psrr_n', 'cmrr']
    vid_db   = db20(xf['VID'].values)
    vicm_db  = db20(xf['VICM'].values)
    vsupp_db = db20(xf['VSUPP'].values)
    vsupn_db = db20(xf['VSUPN'].values)
    psrr_p   = (vid_db - vsupp_db)[0].item()
    psrr_n   = (vid_db - vsupn_db)[0].item()
    cmrr     = (vid_db - vicm_db)[0].item()
    ratios   = pd.DataFrame( np.array([[psrr_p, psrr_n, cmrr]])
                           , columns = cols )
    return ratios

def out_swing_ac( ac: pd.DataFrame, A3dB: float, vdd: float = 1.8
                 ) -> pd.DataFrame:
    cols   = ['v_ih', 'v_il']
    vicm   = ac['vicm'].values
    out_ac = db20(ac['OUT'].values)
    leq_0  = out_ac[vicm <= 0.0]
    geq_0  = out_ac[vicm >= 0.0]
    vil_ac = (vicm[np.argmin(np.abs(leq_0 - A3dB))] + (vdd / 2.0)
             ).real.item() if leq_0.size > 0 else 0.0
    vih_ac = (vicm[np.argmin(np.abs(geq_0 - A3dB))] + (vdd / 2.0)
             ).real.item() if geq_0.size > 0 else 0.0
    swing  = pd.DataFrame(np.array([[vih_ac, vil_ac ]]), columns = cols)
    return swing

def output_current(dc3: pd.DataFrame, dc4: pd.DataFrame) -> pd.DataFrame:
    cols      = ['i_out_min', 'i_out_max']
    i_out_min = dc3['DUT:O'].values[0].item()
    i_out_max = dc4['DUT:O'].values[0].item()
    current   = pd.DataFrame( np.array([[i_out_min, i_out_max]])
                            , columns = cols )
    return current

def operating_point( dcop: pd.DataFrame, dcop_params: dict[str, str]
                   ) -> pd.DataFrame:
    col_ids = { 'DUT:VDD': 'idd', 'DUT:VSS': 'iss'
              } | dcop_params
    dcop    = dcop[list(col_ids.keys())].rename(columns = col_ids)
    return dcop

def estimated_area(expr: str, sizing: pd.DataFrame) -> pd.DataFrame:
    params = to_dict(sizing)
    area   = np.array([[eval(expr, {}, params)]])
    return pd.DataFrame(area, columns = ['area'])
