import os 
import shutil
from typing import List, Union, Dict, NewType, Tuple
import numpy as np
import pickle 
from scipy.interpolate import interp1d
import pandas as pd 
negative_byte = b''

def get_list_of_strings_from_file(filename: str) -> List[str]: 
    # os.path.join(data_path, 'mp_names_saved.txt')
    with open(filename, 'r') as f:
        all_names_str = f.read()
    relevant_mp_columns = all_names_str.split(',')
    return relevant_mp_columns

def make_or_destroy_and_make_dir(dir: str) -> None: 
    if not os.path.exists(dir): 
        os.mkdir(dir)
    else: 
        shutil.rmtree(dir)
        os.mkdir(dir)
    return None


def write_list_to_file_as_string(filename: str, liststrings: List[str]) -> None: 
    line = ','.join(liststrings)
    with open(filename, 'w') as f: 
        f.write(f'{line}')
    
def parse_additional_feature_cols(original_feature_cols: List[str]) -> List[str]:
    parsed_feature_cols = original_feature_cols.copy()
    
    if 'stability_labels' in original_feature_cols: 
        parsed_feature_cols.remove('stability_labels')
        for val in ['label_peeling', 'label_ballooning', 'label_transition']: 
            parsed_feature_cols.append(val)
    return parsed_feature_cols

import aug_sfutils
PULSE_DICT = NewType('AUG_DICT', Dict[str, Dict[str, Dict[str, np.ndarray]]]) 
def get_local_pulse_dict(shot_number: Union[int, str], folder_name: str): 
    filename = os.path.join(folder_name, shot_number)
    with open(filename, 'rb') as file: 
        pulse_dict: PULSE_DICT = pickle.load(file)
    return pulse_dict

def get_local_pulse_arrays(shot_number: Union[int, str], folder_name: str) -> Tuple[np.ndarray]: 
    """ returns (profiles, mps, time, radii) """
    filename = os.path.join(folder_name, shot_number)
    profiles = np.load(filename + '_PROFS.npy')
    mps = np.load(filename + '_MP.npy')
    time = np.load(filename + '_TIME.npy')
    radii = np.load(filename + '_RADII.npy')

    return profiles, mps, time, radii

class NotDesiredShot(Exception): 
    def __init__(self, reason: str, shotno: str) -> None:
        self.shotno = shotno
        self.message = f'{shotno} not desired because {reason}'
        super().__init__(reason, shotno)

class RawPulseDictErrorMissingInformation(Exception): 
    def __init__(self, reason: str, shotno: str) -> None: 
        self.shotno = shotno
        self.message = f'{shotno} not saved because {reason} did not exist'
        super().__init__(reason, shotno)

class ShortPulse(Exception): 
    def __init__(self, shotno: str, total_time: float) -> None: 
        self.shotno = shotno
        self.message = f'{shotno} not saved because only {total_time}s after making array format'
        super().__init__(total_time, shotno)


def check_pulse_dict(pulse_dict: PULSE_DICT, shotno: str, device_name: str='AUG'):
    for key in pulse_dict.keys(): 
        if pulse_dict[key] is None: 
            if key == 'journal' and device_name == 'JET':
                continue 
            else: 
                raise RawPulseDictErrorMissingInformation(key, shotno=shotno)
            
    for key in pulse_dict['profiles'].keys(): 
        if pulse_dict['profiles'][key] is None: 
            raise RawPulseDictErrorMissingInformation(key, shotno=shotno)

    
def check_pulse_arrays(profiles: np.ndarray, mps: np.ndarray, radii: np.ndarray,  times: np.ndarray, shotno: str, mp_names: List[str], device_name='AUG'): 
    if any(len(arr) == 0 for arr in [profiles, mps, times, radii]): 
        raise NotDesiredShot(reason='empty', shotno=shotno)
    if device_name == 'AUG': 
        if times[-1] - times[0] < 4: 
            raise ShortPulse(shotno, times[-1] - times[0])
        for name in ['IpiFP', 'BTF', 'N_tot', 'q95']: 
            mean_mp_val = mps[:, mp_names.index(name)].mean()
            if name in ['IpiFP'] and mean_mp_val < 0: 
                raise NotDesiredShot(reason=f'{name} avg for pulse < 0', shotno=shotno)
            if name in ['BTF', 'N_tot', 'q95'] and mean_mp_val > 0: 
                raise NotDesiredShot(reason=f'{name} avg for pulse > 0', shotno=shotno)
    else: 
        pass 

def select_specific_time_slices(relevant_profiles: np.ndarray, relevant_machine_parameters: np.ndarray, relevant_radii: np.ndarray, relevant_times: np.ndarray, mp_names: List[str], device_name: str): 
    # limit by heating power 
    ptot = np.zeros_like(relevant_times)
    for key in ['PNBI_TOT', 'PICR_TOT', 'PECR_TOT', 'P_OH']:
        ptot += relevant_machine_parameters[:, mp_names.index(key)]
    if device_name == 'AUG': 
        relevant_time_windows_bool: np.array = ptot > 3e6 
    else: 
        relevant_time_windows_bool: np.array = ptot > 2e6

    # Check that q95 is lower than 12 and kappa is positive
    kappa_bool = np.logical_and(abs(relevant_machine_parameters[:, mp_names.index('k')]) > 0.0, abs(relevant_machine_parameters[:, mp_names.index('k')] < 2.0))
    q95_bool = abs(relevant_machine_parameters[:, mp_names.index('q95')]) < 12.0 
    relevant_time_windows_bool = np.logical_and(relevant_time_windows_bool, q95_bool)
    relevant_time_windows_bool = np.logical_and(relevant_time_windows_bool, kappa_bool)

    relevant_profiles = relevant_profiles[relevant_time_windows_bool]
    relevant_radii = relevant_radii[relevant_time_windows_bool]
    relevant_times = relevant_times[relevant_time_windows_bool]
    relevant_machine_parameters = relevant_machine_parameters[relevant_time_windows_bool]

    if len(relevant_profiles) > 10: 
        rand_slices = np.random.permutation(len(relevant_profiles))[:10]
        pressures = np.prod(relevant_profiles, axis=1)* 1.6022e-19 # [:, 0] * relevant_profiles[:, 1] 
        index = np.argmin(np.abs(relevant_radii - 0.85), axis=1)
        index = int(np.mean(index))
        pressure_slices = pressures[:, index]
        pressure_max_slices = np.argpartition(pressure_slices, -3)[-3:]
        select_slices = np.unique(np.concatenate((rand_slices, pressure_max_slices)))
        select_slices_bool = select_slices[relevant_profiles[select_slices, 1, 0] > 2.0e3]
        relevant_times = relevant_times[select_slices_bool]
        relevant_profiles = relevant_profiles[select_slices_bool]
        relevant_radii = relevant_radii[select_slices_bool]
        relevant_machine_parameters = relevant_machine_parameters[select_slices_bool]    
    return relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_times

def example_mappings(profiles, mps, radii, time, device_name: str, shotno):
    profiles_remapped, radii_remapped, mps_remapped, times_remapped = [], [], [], []
    psi_lb, psi_ub, num_points = 0.85, 1.05, 30
    remapx_axis = np.linspace(psi_lb, psi_ub, num_points)
    if device_name == 'AUG': 
        radii = radii**2 # AUG DATA comes in RHO
    removed_slices = 0
    for slice_idx in range(profiles.shape[0]): 
        try:
            radval = radii[slice_idx]
            teval = profiles[slice_idx, 1]
            neval =  profiles[slice_idx, 0]
            ind = np.where(neval < 5e18)
            ind1 = np.where(teval > 50)
            index = np.intersect1d(ind,ind1)
            teval[index] = neval[index]*50/5e18
            ind = np.where(neval<1e19)
            ind1 = np.where(teval*neval*1.6022e-19 > 400)
            index1 = np.intersect1d(ind,ind1)
            teval[index1] = neval[index1]*100/1e19
            f_te = interp1d(radval, teval)
            xrtest = np.linspace(0.85,1.05,1000)
            tevs = f_te(xrtest)
            index1 = np.where(tevs - 100 < 0)
            psinsep = xrtest[index1[0][0]]
            radval = radval + (1.0 - psinsep)
            f_ne = interp1d(radval, neval)
            ne_remap = f_ne(remapx_axis)
            f_te = interp1d(radval, teval)
            te_remap = f_te(remapx_axis)
        
            ind1 = np.where(remapx_axis > 1.0)
            ind2 = np.where(te_remap > 150)
            index1 = np.intersect1d(ind1,ind2)
            te_remap[index1] = 100*np.exp(-(remapx_axis[index1] - 1.0)/0.03) 
            # pe_remap = eV*np.multiply(ne_remap, te_remap)
            # prof_remap = np.stack([ne_remap, te_remap, pe_remap], axis=0)
            prof_remap = np.stack([ne_remap, te_remap], axis=0)
            profiles_remapped.append(prof_remap)
            radii_remapped.append(remapx_axis)
            mps_remapped.append(mps[slice_idx])
            times_remapped.append(time[slice_idx])

        except ValueError: 
            # shot_num = shot.split('/')[-1]
            # print(f'Slice {slice_idx} in {shot_num} (time = {time[slice_idx]}) falls outside of the interpolation range of {psi_lb}, {psi_ub}: MAX {radii[slice_idx].max():.4}, MIN {radii[slice_idx].min():.5}')
            removed_slices += 1
            continue 
        except IndexError: 
            removed_slices += 1
            continue 

    if len(profiles_remapped) == 0: 
        raise NotDesiredShot(reason='not enough profiles after remapping', shotno=shotno)
    else: 
        return np.stack(profiles_remapped,0), np.stack(mps_remapped,0), np.stack(times_remapped,0), np.stack(radii_remapped,0)
    
def map_pulse_dict_to_numpy_arrays(pulse_dict: PULSE_DICT, filter_time_by: str, relevant_mps_columns: List[str], device_name:str='AUG', **kwargs) -> List[np.ndarray]: 

    profile_data, mp_data = pulse_dict['profiles'], pulse_dict['machine_parameters']
    if mp_data.get('PECR_TOT', None) is None: 
        mp_data['PECR_TOT'] = {}
        mp_data['PECR_TOT']['time'], mp_data['PECR_TOT']['data'] = 'NO ECRH USED', None
    ida_times, ne, te, radius = profile_data['time'], profile_data['ne'], profile_data['Te'], profile_data['radius']
    if ne.shape[0] == 200: 
        # get time on x-axis for AUG pulses
        ne, te, radius = ne.T, te.T, radius.T
    profiles = np.stack((ne, te), 1)

    avail_cols = [key for key in relevant_mps_columns if key in mp_data.keys()]

    MP_TIME_LIST = np.array([(min(mp_data[key]['time']), max(mp_data[key]['time'])) for key in avail_cols if isinstance(mp_data[key], dict) and not isinstance(mp_data[key]['time'], str) and mp_data.get(key) is not None])
    largest_mp_start_time, smallest_mp_end_time = MP_TIME_LIST.max(axis=0)[0], MP_TIME_LIST.min(axis=0)[1]
    smallest_ida_end_time, largest_ida_start_time = ida_times[-1], ida_times[0]
    
    if filter_time_by == 'both': 
        t1, t2 = max(largest_mp_start_time, largest_ida_start_time), min(smallest_mp_end_time, smallest_ida_end_time)
    elif filter_time_by == 'ida': 
        t1, t2 = largest_ida_start_time, smallest_ida_end_time
    elif filter_time_by == 'flattop': 
        if device_name == 'AUG': 
            t1, t2 = pulse_dict['journal']['flatb'], pulse_dict['journal']['flate']
        else: 
            raise NotImplementedError
    
    relevant_time_windows_bool: np.array = np.logical_and(ida_times > t1, ida_times < t2)
    
    relevant_time_windows: np.array = ida_times[relevant_time_windows_bool]
    relevant_profiles = profiles[relevant_time_windows_bool]
    relevant_radii = radius[relevant_time_windows_bool]

    relevant_machine_parameters: np.array = np.empty((len(relevant_profiles), len(relevant_mps_columns)))
    for mp_idx, key in enumerate(relevant_mps_columns): 
        relevant_mp_vals = np.zeros(len(relevant_profiles))
        if not mp_data.get(key): # check for key! 
            mp_raw_data, mp_raw_time = None, None
        else: 
            mp_raw_data, mp_raw_time = mp_data[key]['data'], mp_data[key]['time']
        if mp_raw_time is None or isinstance(mp_raw_time, str): # this catches whenever NBI isn't working or the string in JET pulse files 'NO_ICRH_USED'
            pass 
        elif len(mp_raw_data) != len(mp_raw_time): 
            raise RawPulseDictErrorMissingInformation(f'MP {key} does not have equal data and time lengths', 00000)
        else:
            f = interp1d(mp_raw_time, mp_raw_data, bounds_error=False, fill_value=0.0)
            relevant_mp_vals = f(relevant_time_windows)
        if key in ['D_tot', 'N_tot']: 
            relevant_mp_vals*=1e-22
            relevant_mp_vals = np.where(relevant_mp_vals < 1e-5, 0.0, relevant_mp_vals)# np.clip(relevant_mp_vals, a_min=1e-5, a_max=None)
        if key == 'tau_tot': 
            relevant_mp_vals = np.clip(np.nan_to_num(relevant_mp_vals), a_min=1e-5, a_max=3)
        # if key in ['Rgeo', 'ahor']: 
        #     relevant_mp_vals = np.clip(relevant_mp_vals, a_min=1e-5, a_max=None)
        # if key in ['IpiFP']: 
        #     relevant_mp_vals = abs(relevant_mp_vals)
        relevant_machine_parameters[:, mp_idx] = relevant_mp_vals
    return relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_time_windows, relevant_mps_columns

def precompute_combined_features(additional_feature_engineering_cols: List[str], relevant_mp_columns:List[str], relevant_profiles: np.ndarray, relevant_machine_parameters: np.ndarray, relevant_radii: np.ndarray, relevant_times: np.ndarray) -> dict: 
    computed_features = {}
    if any(item in additional_feature_engineering_cols for item in ['stability_norm', 'stability_alpha', 'stability_jb', 'stability_ratio']):
        alpha_maxes, jb_maxes, norm_mag, thetas, alpha_normalized_pressure_gradient, bootstrap = calculate_stability_profiles_and_maxes_for_pulse(relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_times, relevant_mp_columns)
        computed_features = dict(**computed_features, alpha_maxes=alpha_maxes, jb_maxes=jb_maxes, norm_mag=norm_mag, thetas=thetas, alpha_normalized_pressure_gradient=alpha_normalized_pressure_gradient, bootstrap=bootstrap)
    if any(item in additional_feature_engineering_cols for item in ['label_peeling', 'label_ballooning', 'label_transition', 'stability_ratio']): 
        plasma_current = relevant_machine_parameters[:, relevant_mp_columns.index('IpiFP')]        
        plh_ratio = calculate_plh_ratio(relevant_profiles, relevant_machine_parameters, relevant_mp_columns)
        stability_labels, stability_ratios = calculate_stability_labels(thetas, norm_mag, alpha_maxes, jb_maxes, plasma_current, plh_ratio, relevant_times)
        computed_features = dict(**computed_features, stability_labels=stability_labels, stability_ratios=stability_ratios)
    return computed_features

def pre_filter_aug_pulses_from_journal(pulse_dict: PULSE_DICT, shot_num: str) -> None: 
    # Grab only h-mode, deterium fuelled, non-disruptive plasmas
    filters = ['b_hmod', 'gas_d', 'b_disr'] # TODO: move outside to script
    for filter in filters:     
        if filter in ['b_hmod', 'gas_d']: 
            if pulse_dict['journal'][filter] == negative_byte: 
                raise NotDesiredShot(filter, shot_num)
        elif filter in ['b_disr']: 
            if pulse_dict['journal'][filter] != negative_byte: 
                raise NotDesiredShot(filter, shot_num)

def pre_filter_pulses(pulse_dict: PULSE_DICT, shot_num: str, device_name: str, **kwargs) -> None: 
    if device_name == 'AUG': 
        pre_filter_aug_pulses_from_journal(pulse_dict, shot_num)
    else: 
        jet_pdb = kwargs.get('jet_pdb')
        shot_loc_pdb = jet_pdb[jet_pdb['shot'] == int(shot_num)].iloc[0]
        if int(shot_num) < 81000: 
            raise NotDesiredShot('JET-C pulse', shot_num)
        if shot_loc_pdb['flowrateofseededimpurity10^22(e/s)'] > 0.05e22: 
            raise NotDesiredShot('Impurity seeded', shot_num)
        if shot_loc_pdb['FLAG:HYDROGEN'] > 0.0 or shot_loc_pdb['FLAG:HeJET-C'] > 0.0: 
            raise NotDesiredShot('Not pure deterium', shot_num)
        if shot_loc_pdb['FLAG:Kicks'] > 0.0 or shot_loc_pdb['FLAG:RMP'] > 0.0 or shot_loc_pdb['FLAG:pellets'] > 0.0:
            raise NotDesiredShot('Kicks, RMPs, or pellets', shot_num)

def get_jetpdb(filename: str): 
    df = pd.read_csv(filename)
    return df 
from physics import calculate_plh_ratio, calculate_stability_profiles_and_maxes_for_pulse, calculate_stability_labels
def map_additional_feature(feature: str, n_idx: int, relevant_mp_columns: List[str], relevant_profiles: np.ndarray, relevant_machine_parameters: np.ndarray, relevant_radii: np.ndarray, relevant_times: np.ndarray, pulse_dict: PULSE_DICT, precomputed_features:dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
    if feature == 'aspect_ratio': 
        # major radius / minor radius 
        r_idx, a_idx = relevant_mp_columns.index('Rgeo'), relevant_mp_columns.index('ahor')
        relevant_machine_parameters[:, n_idx] = relevant_machine_parameters[:, r_idx] / relevant_machine_parameters[:, a_idx]
    elif feature == 'inverse_aspect_ratio': 
        # minor radius / major radius
        r_idx, a_idx = relevant_mp_columns.index('Rgeo'), relevant_mp_columns.index('ahor')
        relevant_machine_parameters[:, n_idx] = relevant_machine_parameters[:, a_idx] / relevant_machine_parameters[:, r_idx]
    elif feature == 'P_TOT/P_LH':
        plh_ratio = calculate_plh_ratio(relevant_profiles, relevant_machine_parameters, relevant_mp_columns)
        plh_ratio = np.clip(plh_ratio, a_min=0, a_max=20)
        relevant_machine_parameters[:, n_idx] = plh_ratio
    elif feature == 'P_TOT': 
        ptot = np.zeros(len(relevant_machine_parameters))
        for key in ['PNBI_TOT', 'PICR_TOT','PECR_TOT', 'P_OH']: 
            rel_pow_col_idx = relevant_mp_columns.index(key)
            ptot += relevant_machine_parameters[:, rel_pow_col_idx]
        ptot = np.clip(ptot, a_min=0.0, a_max=1e9)
        relevant_machine_parameters[:, n_idx] = ptot
    elif feature == 'stability_norm':
        relevant_machine_parameters[:, n_idx] = precomputed_features.get('norm_mag', None)
    elif feature == 'stability_alpha':
        relevant_machine_parameters[:, n_idx] = precomputed_features.get('alpha_maxes', None)
    elif feature == 'stability_jb':
        relevant_machine_parameters[:, n_idx] = precomputed_features.get('jb_maxes', None)
    elif feature == 'stability_thetas': 
        relevant_machine_parameters[:, n_idx] = precomputed_features.get('thetas', None) 
    elif feature == 'stability_ratio': 
        relevant_machine_parameters[:, n_idx] = precomputed_features.get('stability_ratios', None) 
    elif feature == 'label_peeling': 
        stability_labels = precomputed_features.get('stability_ratios', None)
        vals = np.logical_or(np.where(stability_labels == 2, 1, 0), np.where(stability_labels == 3, 1, 0)).astype(int)
        relevant_machine_parameters[:, n_idx] = vals
    elif feature == 'label_ballooning': 
        stability_labels = precomputed_features.get('stability_ratios', None)
        vals = np.logical_or(np.where(stability_labels == 1, 1, 0), np.where(stability_labels == 3, 1, 0)).astype(int)
        relevant_machine_parameters[:, n_idx] = vals
    elif feature == 'label_transition': 
        stability_labels = precomputed_features.get('stability_ratios', None)
        vals = np.where(stability_labels == 0, 1, 0)
        relevant_machine_parameters[:, n_idx] = vals
    elif feature == 'impurity_gas': 
        val = np.zeros(len(relevant_machine_parameters))
        impuritykeys = ['gas_he', 'gas_ne', 'gas_ar', 'gas_n2', 'gas_kr', 'gas_xe', 'gas_cd4', 'gas_other']
        for key in impuritykeys: 
            if pulse_dict['journal'][key] != negative_byte:
                val = val + impuritykeys.index(key) + 1
                break
        relevant_machine_parameters[:, n_idx] = val
    else: 
        raise NotImplementedError(f'Feature transformation not defined for {feature}')
        
    return relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_times