""" Single-Ended Operational Amplifier Characterization Utility functions """

from   tempfile        import mkdtemp
from   shutil          import copyfile
from   typing          import Union
from   collections.abc import Iterable
import os
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
    return np.argmin(np.abs(array - value))

def db20(x: Union[float, np.array]) -> Union[float, np.array]:
    return np.log10(np.abs(x)) * 20.0

def nan_frame(cols: Iterable[str]) -> pd.DataFrame:
    length = len(cols)
    return pd.DataFrame(np.full((1,length), np.NaN), columns = cols)
