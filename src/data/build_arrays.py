import argparse
from typing import List, Union, Tuple
import os 
from multiprocessing import Pool
import psutil
import numpy as np
import _utils as datautils
import yaml 
import sys 
import logging 
# import datautils

negative_byte = b''


def make_arrays(shot_num): 
    pulse_dict = datautils.get_local_pulse_dict(shot_num, args.raw_folder_name)

    # initial feature filtering which raises exceptions based on the filtering
    datautils.pre_filter_aug_pulses_from_journal(pulse_dict, shot_num, args.config['feature_filtering'])
    # filter pulses with any missing data
    datautils.filter_pulses_with_missing_data(pulse_dict, shot_num)

    # map the pulses to dictionary
    relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_times, mp_names = map_pulse_dict_to_numpy_arrays(pulse_dict, relevant_mps_columns=relevant_mp_columns)
    datautils.check_pulse_arrays(relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_times, shot_num, mp_names)

    additional_feature_cols = np.empty((len(relevant_profiles), len(args.config['additional_feature_engineering'])))
    relevant_machine_parameters = np.concatenate([relevant_machine_parameters, additional_feature_cols], 1)
    precomputed_features = datautils.precompute_combined_features(args.config['additional_feature_engineering'], relevant_mp_columns, relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_times)
    for n_idx, feature in enumerate(args.config['additional_feature_engineering'], start=len(relevant_mp_columns)):
        relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_times = datautils.map_additional_feature(feature, n_idx, relevant_mp_columns, relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_times, precomputed_features, pulse_dict)

    return relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_times

from scipy.interpolate import interp1d
def map_pulse_dict_to_numpy_arrays(pulse_dict: datautils.PULSE_DICT, relevant_mps_columns: List[str], device_name:str='AUG', **kwargs) -> List[np.ndarray]: 
    profile_data, mp_data = pulse_dict['profiles'], pulse_dict['machine_parameters']
    if mp_data.get('PECR_TOT', None) is None: 
        mp_data['PECR_TOT'] = {}
        mp_data['PECR_TOT']['time'], mp_data['PECR_TOT']['data'] = 'NO ECRH USED', None
    ida_times, ne, te, radius = profile_data['time'], profile_data['ne'], profile_data['Te'], profile_data['radius']
    if ne.shape[0] == 200: 
        # get time on 1st dimension of array for AUG pulses
        ne, te, radius = ne.T, te.T, radius.T
    profiles = np.stack((ne, te), 1)

    avail_cols = [key for key in relevant_mps_columns if key in mp_data.keys()]

    MP_TIME_LIST = np.array([(min(mp_data[key]['time']), max(mp_data[key]['time'])) for key in avail_cols if isinstance(mp_data[key], dict) and not isinstance(mp_data[key]['time'], str) and mp_data.get(key) is not None])
    largest_mp_start_time, smallest_mp_end_time = MP_TIME_LIST.max(axis=0)[0], MP_TIME_LIST.min(axis=0)[1]
    smallest_ida_end_time, largest_ida_start_time = ida_times[-1], ida_times[0]
    
    t1, t2 = largest_ida_start_time, smallest_ida_end_time

    # ! past work on filtering time..
    # ! if filter_time_by == 'both': 
    # !     t1, t2 = max(largest_mp_start_time, largest_ida_start_time), min(smallest_mp_end_time, smallest_ida_end_time)
    # ! elif filter_time_by == 'ida': 
    # !     t1, t2 = largest_ida_start_time, smallest_ida_end_time
    # ! elif filter_time_by == 'flattop': 
    # !     if device_name == 'AUG': 
    # !         t1, t2 = pulse_dict['journal']['flatb'], pulse_dict['journal']['flate']
    # !     else: 
    # !         raise NotImplementedError
    
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
        f = interp1d(mp_raw_time, mp_raw_data, bounds_error=False, fill_value=(mp_raw_data[0], mp_raw_data[-1]))
        relevant_mp_vals = f(relevant_time_windows)
        relevant_machine_parameters[:, mp_idx] = relevant_mp_vals
    return relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_time_windows, relevant_mps_columns

# !-------------------------------------------------
# ! Post-Processing after mapping functions
# !-------------------------------------------------

def post_process_machine_parameters(mps: np.ndarray, config: dict) -> np.ndarray: 
    """ 
    1. D_tot are measured in e/s, 
        therefore the order of magnitude is ~1e22, 
        so we scale it down, and clip anything that is too low to be registered
    """
    for key in ['D_tot']: 
        mp_idx = config['all_mps_cols'].index(key)
        mps[:, mp_idx] *= 1e-22
        mps[:, mp_idx] = np.where(mps[:, mp_idx] < 1e-5, 0.0, mps[:, mp_idx])
    return mps

# !-------------------------------------------------
# ! Anomaly detection and updating!
# !-------------------------------------------------

def anomaly_detection_profiles(profiles: np.ndarray, config: dict, shotno: str) -> np.ndarray: 
    # TODO: a lot
    ne, te = profiles[:, 0]*1e-19, profiles[:, 1]*1e-3
    # ! if there are more than 30 slices which have an change in profile average 
    # ! of density/temperature that is > 3stds 
    # ! of the average change of the averaged profile over whole pulse
    # ! and the change in profile average is > 0.5 (normalized to keV and 1e-19)
    # ! discard the pulse 
    if (ne > 40).sum() > 0: 
        raise datautils.ProfileAnomaly(reason=f'Density greater than 4e20... dropping', shotno=shotno)
    if (te > 20).sum() > 0: 
        raise datautils.ProfileAnomaly(reason=f'Temperature greater than 20keV... dropping', shotno=shotno)
    ne_bool = np.logical_and(np.gradient(ne.mean(axis=1)) > np.gradient(ne.mean(axis=1)).mean() + 3*np.gradient(ne.mean(axis=1)).std(), np.gradient(ne.mean(axis=1)) > 0.5)
    # TODO: check against the gas injection! 
    if ne_bool.sum() > 30: 
        raise datautils.ProfileAnomaly(reason=f'Density - {(ne_bool).sum()} slices with higher than 3 sigma deviation', shotno=shotno)
    # TODO: check against the power injection! 
    te_bool = np.logical_and(np.gradient(te.mean(axis=1)) > np.gradient(te.mean(axis=1)).mean() + 3*np.gradient(te.mean(axis=1)).std(), np.gradient(te.mean(axis=1)) > 0.5)
    if te_bool.sum() > 30: 
        raise datautils.ProfileAnomaly(reason=f'Temperature - {(te_bool).sum()} slices with higher than 3 sigma deviation', shotno=shotno)
    return profiles

def anomaly_detection_machine_parameters(mps: np.ndarray, config: dict, shotno: str) -> Tuple[np.ndarray, bool]: 
    """ 
    """
    # check RGEO and AHOR
    retriggering = False
    for key in ['Rgeo', 'ahor']: 
        mp_idx = config['all_mps_cols'].index(key)
        clamp_val = 1e-4
        if mps[:, mp_idx].min() < clamp_val: 
            logging.warning(f'{shotno} has very low values of {key}, clamping to {clamp_val}')
            mps[:, mp_idx] = np.clip(mps[:, mp_idx], a_min=clamp_val, a_max=None)
            # if the rgeo has changed then you have to change aspect ratio as well... 
            retriggering = True
            if 'aspect_ratio' in config['all_mps_cols']: 
                aspect_idx, r_idx, a_idx = config['all_mps_cols'].index('aspect_ratio'), config['all_mps_cols'].index('Rgeo'), config['all_mps_cols'].index('ahor')
                mps[:, aspect_idx] = mps[:, r_idx] / mps[:, a_idx]
            if 'inverse_aspect_ratio' in config['all_mps_cols']: 
                aspect_idx, r_idx, a_idx = config['all_mps_cols'].index('aspect_ratio'), config['all_mps_cols'].index('Rgeo'), config['all_mps_cols'].index('ahor')
                mps[:, aspect_idx] =  mps[:, a_idx] / mps[:, r_idx]
            # TODO: retrigger the whole map additional features? 
            # possibly also the stability stuffs... 

    for key in ['tau_tot']: 
        mp_idx = config['all_mps_cols'].index(key)
        min_val, max_val = 1e-5, 3
        if np.isnan(mps[:, mp_idx]).any(): 
            retriggering = True
            logging.warning(f'{shotno} has nans, setting to Nans zero, neginf to large neg. value, and posinf to large pos. value')
            mps[:, mp_idx] = np.nan_to_num(mps[:, mp_idx], nan=0.0, neginf=-100000, posinf=1000000)
        if mps[:, mp_idx].min() < min_val or mps[:, mp_idx].max() > max_val: 
            logging.warning(f'{shotno} has {key} lower than {min_val} and higher than {max_val}')
            mps[:, mp_idx] = np.clip(mps[:, mp_idx], a_min=min_val, a_max=max_val)
            retriggering = True
    
    for key in ['PNBI_TOT', 'PECR_TOT', 'PICR_TOT']:
        mp_idx = config['all_mps_cols'].index(key)
        if mps[:, mp_idx].min() < 0.0:  
            raise MPAnomaly(reason=f'Negative values of {key}', shotno=shotno)
    # if key in ['IpiFP']: 
    #     relevant_mp_vals = abs(relevant_mp_vals)

    # TODO: Check  for rapid changes in machine parameters...
    return mps, retriggering


def save_arrays(save_dir: str, profiles: np.ndarray, mps: np.ndarray, radii: np.ndarray, times: np.ndarray, shotno: Union[int, str]): 
    def save_array(path: str, _data_name: str, arr: np.ndarray): 
        with open(path + f'_{_data_name}.npy', 'wb') as file: 
            np.save(file, arr)
    relevant_path = os.path.join(save_dir, shotno)
    for data_name, data in zip(['PROFS', 'MP', 'RADII', 'TIME'], [profiles, mps, radii, times]): 
        save_array(relevant_path, data_name, data)
    print(f'{shotno} saved to {relevant_path}')
    logging.info(f'{shotno} saved to {relevant_path}')

def build(shot_num: str): 
    try: 
        profiles, mps, radii, times = make_arrays(shot_num)
        mps = post_process_machine_parameters(mps, args.config)
        mps, retrigger_features = anomaly_detection_machine_parameters(mps, args.config, shot_num)
        profiles = anomaly_detection_profiles(profiles, args.config, shot_num)
        # TODO: mapping functions
        # TODO: if anomaly detected, then retrigger below
        if retrigger_features:
            precomputed_features = datautils.precompute_combined_features(args.config['additional_feature_engineering'], \
                                                                        relevant_mp_columns, profiles, mps, radii, times)

            for n_idx, feature in enumerate(args.config['additional_feature_engineering'], start=len(relevant_mp_columns)):
                profiles, mps, radii, times = datautils.map_additional_feature(feature, n_idx, relevant_mp_columns, profiles, mps, radii, times, precomputed_features)
    # TODO: change below to high level aerrors, anomoly, not desired ,missing data, 
    except datautils.NotDesiredShot as e: 
        logging.info(e.message)
        # logging.info(f'Shot #{e.shotno} not desired because {e.reason}')
    except datautils.RawPulseDictErrorMissingInformation as e: 
        logging.warning(e.message)
        # logging.warning(f'Shot #{e.shotno} missing data: {e.reason}')
    except datautils.ShortPulse as e: 
        logging.info(e.message)
        # logging.info(f'Shot #{e.shotno} deemed too short: {e.total_time}')
    except datautils.ProfileAnomaly as e: 
        logging.warning(e.message)
    except RuntimeWarning as e: 
        logging.error(f'Shot #{shot_num} has unexpected error: {e}')
    except datautils.MPAnomaly as e: 
        logging.warning(e.message)
    else: 
        save_arrays(SAVE_DIR, profiles, mps, radii, times, shot_num)
        logging.info(f'Saved shot #{shot_num}')



def log_config_parameters(config, parent_key=''):
    for key, value in config.items():
        full_key = f'{parent_key}.{key}' if parent_key else key
        if isinstance(value, dict):
            log_config_parameters(value, full_key)
        else:
            logging.info(f'Config parameter: {full_key} = {value}')

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Process raw data into numpy arrays')
    parser.add_argument('-rf', '--raw_folder_name', type=str, default=None)
    parser.add_argument('-lf', '--array_folder_name', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('-mp', action='store_true', default=False, help='To do multiprocessing or not')
    args = parser.parse_args()

    log_name = args.config.split('/')[-1].split('.yaml')[0]
    logging.basicConfig(filename=f'build_logs_{log_name}.log', format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, filemode='w')

    config = datautils.read_yaml_input_file(fname=args.config)

    # log args and config 
    logging.info(f"Raw Data Folder Location: {args.raw_folder_name}")
    logging.info(f"Processed Arrays storage Location: {args.array_folder_name}")
        
    relevant_mp_columns = ['BTF', 'IpiFP', 'D_tot', 'N_tot', 'P_OH', 'PNBI_TOT', 'PICR_TOT', 'PECR_TOT','SHINE_TH', 'k', 'delRoben', 'delRuntn', 'ahor', 'Rgeo', 'q95', 'Vol', 'Wmhd', 'Wfi', 'Wth', 'dWmhd/dt', 'tau_tot']
    config['all_mps_cols'] = relevant_mp_columns+config['additional_feature_engineering']
    SAVE_DIR = args.array_folder_name
    datautils.make_or_destroy_and_make_dir(SAVE_DIR)
    datautils.write_list_to_file_as_string(filename=os.path.join(SAVE_DIR, 'mp_names_saved.txt'), liststrings=config['all_mps_cols'])

    shot_list = sorted(os.listdir(args.raw_folder_name))
    logging.info(f'Building Arrays based on {len(shot_list)} pulses found in {args.raw_folder_name}')
    args.config = config 
    log_config_parameters(args.config)
    
    if args.mp: 
        with Pool(psutil.cpu_count(logical=False) -2) as pool: 
            pool.map(build, shot_list)
    else: 
        for shot in shot_list: 
            build(shot)

    logging.info(f'Total Shots Saved {(len(os.listdir(args.array_folder_name)) - 1 )// 4}')

    