import argparse
from typing import List, Union
import os 
from multiprocessing import Pool
import psutil
import numpy as np
import _utils as datautils
import yaml 
import sys 
import logging 
import data_exceptions

negative_byte = b''

def make_arrays(shot_num): 
    pulse_dict = datautils.get_local_pulse_dict(shot_num, args.raw_folder_name)

    # initial feature filtering which raises exceptions based on the filtering
    datautils.pre_filter_aug_pulses_from_journal(pulse_dict, shot_num, args.config['feature_filtering'])
    # filter pulses with any missing data
    datautils.filter_pulses_with_missing_data(pulse_dict, shot_num)
    # map the pulses to dictionary
    relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_times, mp_names = datautils.map_pulse_dict_to_numpy_arrays(pulse_dict, filter_time_by='ida', relevant_mps_columns=relevant_mp_columns)
    datautils.check_pulse_arrays(relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_times, shot_num, mp_names)

    additional_feature_cols = np.empty((len(relevant_profiles), len(args.config['additional_feature_engineering'])))
    relevant_machine_parameters = np.concatenate([relevant_machine_parameters, additional_feature_cols], 1)

    precomputed_features = datautils.precompute_combined_features(args.config['additional_feature_engineering'], relevant_mp_columns, relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_times)
    for n_idx, feature in enumerate(args.config['additional_feature_engineering'], start=len(relevant_mp_columns)):
        relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_times = datautils.map_additional_feature(feature, n_idx, relevant_mp_columns, relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_times, pulse_dict, precomputed_features)
    return relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_times

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
    except data_exceptions.NotDesiredShot as e: 
        logging.warning(f'Shot #{e.shotno} not desired because {e.reason}')
    except data_exceptions.RawPulseDictErrorMissingInformation as e: 
        logging.warning(f'Shot #{e.shotno} missing data: {e.reason}')
    except data_exceptions.ShortPulse as e: 
        logging.warning(f'Shot #{e.shotno} deemed too short: {e.total_time}')
    except RuntimeWarning as e: 
        logging.error(f'Shot #{shot_num} has unexpected error: {e}')
    else: 
        save_arrays(SAVE_DIR, profiles, mps, radii, times, shot_num)
        logging.info(f'Saved shot #{shot_num}')
    # TODO: Anomoly detection 

import yaml 
def read_yaml_input_file(fname: str) -> dict: 
    with open(fname, 'r') as f: 
        return yaml.safe_load(f)


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

    config = read_yaml_input_file(fname=args.config)
    args.config = config 
    # log args and config 
    logging.info(f"Raw Data Folder Location: {args.raw_folder_name}")
    logging.info(f"Processed Arrays storage Location: {args.array_folder_name}")
    log_config_parameters(args.config)
    
    relevant_mp_columns = ['BTF', 'IpiFP', 'D_tot', 'N_tot', 'P_OH', 'PNBI_TOT', 'PICR_TOT', 'PECR_TOT','SHINE_TH', 'k', 'delRoben', 'delRuntn', 'ahor', 'Rgeo', 'q95', 'Vol', 'Wmhd', 'Wfi', 'Wth', 'dWmhd/dt', 'tau_tot']

    # additional_feature_engineering_cols: List[str] = config['additional_feature_engineering']
    # mapping_functions = config['additional_mapping_functions']
    
    # additional_feature_filtering = config['feature_filtering']
    SAVE_DIR = args.array_folder_name
    datautils.make_or_destroy_and_make_dir(SAVE_DIR)
    # additional_feature_engineering_cols = datautils.parse_additional_feature_cols(config['additional_feature_engineering'])
    datautils.write_list_to_file_as_string(filename=os.path.join(SAVE_DIR, 'mp_names_saved.txt'), liststrings=relevant_mp_columns+config['additional_feature_engineering'])

    shot_list = sorted(os.listdir(args.raw_folder_name))
    logging.info(f'Building Arrays based on {len(shot_list)} pulses found in {args.raw_folder_name}')
    
    
    if args.mp: 
        with Pool(psutil.cpu_count(logical=False) -2) as pool: 
            pool.map(build, shot_list)
    else: 
        for shot in shot_list: 
            build(shot)

    logging.info(f'Total Shots Saved {len(os.listdir())}')

    