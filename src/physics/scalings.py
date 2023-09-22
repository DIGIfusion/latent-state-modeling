import numpy as np 
from typing import List
def calculate_plh_ratio(relevant_profiles: np.ndarray, relevant_machine_parameters: np.ndarray, relevant_mp_columns: List[str]) -> np.ndarray:
    power_scaling = lambda n, r, a, b : (2.14)*(np.exp(0.107))*(np.power(n*1e-20,0.728))*(np.power(abs(b),0.772))*(np.power(a,0.975))*(np.power(r,0.999))
    n = relevant_profiles[:,0,0]
    r = relevant_machine_parameters[:,relevant_mp_columns.index('Rgeo')]
    a = relevant_machine_parameters[:,relevant_mp_columns.index('ahor')]
    b = abs(relevant_machine_parameters[:,relevant_mp_columns.index('BTF')])
    pmartin = power_scaling(n, r, a, b)
    ptot = np.zeros(len(relevant_machine_parameters))
    for key in ['PNBI_TOT', 'PICR_TOT','PECR_TOT', 'P_OH']: 
        if key not in relevant_mp_columns: 
            continue 
        rel_pow_col_idx = relevant_mp_columns.index(key)
        ptot += relevant_machine_parameters[:, rel_pow_col_idx]
    ptot = np.clip(ptot, a_min=0.0, a_max=1e9)
    pmartin = np.clip(1e6*pmartin, a_min=0.01, a_max=1e9) 
    # plh_ratio = ptot / pmartin
    return ptot / pmartin