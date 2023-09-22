import argparse
from typing import List, Union
import os 
from multiprocessing import Pool
import psutil
import numpy as np
import _utils as datautils


def make_arrays(shot_num, folder_name, device_name):
    pulse_dict = datautils.get_local_pulse_dict(shot_num, folder_name)
    datautils.pre_filter_pulses(pulse_dict, shot_num, device_name, jet_pdb=jet_pdb)
    datautils.check_pulse_dict(pulse_dict, shot_num, device_name)
    relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_times, mp_names = datautils.map_pulse_dict_to_numpy_arrays(pulse_dict, filter_time_by='ida', relevant_mps_columns=relevant_mp_columns, device_name=device_name)
    relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_times = datautils.select_specific_time_slices(relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_times, mp_names, device_name)
    datautils.check_pulse_arrays(relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_times, shot_num, mp_names, device_name)
    additional_feature_cols = np.empty((len(relevant_profiles), len(additional_feature_engineering_cols)))
    relevant_machine_parameters = np.concatenate([relevant_machine_parameters, additional_feature_cols], 1)

    precomputed_features = datautils.precompute_combined_features(additional_feature_engineering_cols, relevant_mp_columns, relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_times)
    for n_idx, feature in enumerate(additional_feature_engineering_cols, start=len(relevant_mp_columns)):
        relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_times = datautils.map_additional_feature(feature, n_idx, relevant_mp_columns, relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_times, pulse_dict, precomputed_features)
    relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_times = datautils.example_mappings(relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_times, device_name, shot_num) 
    return relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_times
    
def build(build_args): 
    shot_num, folder_name, device_name = build_args 
    try: 
        profiles, mps, radii, times = make_arrays(shot_num, folder_name, device_name)
    except datautils.NotDesiredShot as e: 
        print(e.message)
    except datautils.RawPulseDictErrorMissingInformation as e: 
        print(e.message)
    except datautils.ShortPulse as e: 
        print(e.message)
    except RuntimeWarning as e: 
        print(e, shot_num)
    else: 
        save_arrays(SAVE_DIR, profiles, mps, radii, times, shot_num)

def save_arrays(save_dir: str, profiles: np.ndarray, mps: np.ndarray, radii: np.ndarray, times: np.ndarray, shotno: Union[int, str]): 
    def save_array(path: str, _data_name: str, arr: np.ndarray): 
        with open(path + f'_{_data_name}.npy', 'wb') as file: 
            np.save(file, arr)
    print(save_dir, shotno)
    relevant_path = os.path.join(save_dir, shotno)
    for data_name, data in zip(['PROFS', 'MP', 'RADII', 'TIME'], [profiles, mps, radii, times]): 
        save_array(relevant_path, data_name, data)
    print(f'{shotno} saved to {relevant_path}')

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Process raw data into numpy arrays')
    parser.add_argument('-rf_aug', '--raw_folder_name_aug', type=str, default=None)
    parser.add_argument('-rf_jet', '--raw_folder_name_jet', type=str, default=None)
    parser.add_argument('-jetpdb_filename', type=str, default=None)
    parser.add_argument('-lf', '--array_folder_name', type=str, required=True)
    parser.add_argument('-feature', '--additional_feature_engineering', action='append', type=str, default=[])
    parser.add_argument('-mf', '--additional_mapping_functions', action='append', default=[], type=str)
    parser.add_argument('-ff', '--additional_filtering_functions', action='append', default=[], type=str)
    parser.add_argument('-mp', action='store_true', default=False, help='To do multiprocessing or not')
    args = parser.parse_args()

    relevant_mp_columns = ['BTF', 'IpiFP', 'D_tot', 'N_tot', 'P_OH', 'PNBI_TOT', 'PICR_TOT', 'PECR_TOT','SHINE_TH', 'k', 'delRoben', 'delRuntn', 'ahor', 'Rgeo', 'q95', 'Vol', 'Wmhd', 'Wfi', 'Wth', 'dWmhd/dt', 'tau_tot']

    additional_feature_engineering_cols: List[str] = args.additional_feature_engineering
    mapping_functions = args.additional_mapping_functions
    additional_feature_filtering = args.additional_filtering_functions
    SAVE_DIR = args.array_folder_name
    
    datautils.make_or_destroy_and_make_dir(SAVE_DIR)
    additional_feature_engineering_cols = datautils.parse_additional_feature_cols(additional_feature_engineering_cols)
    datautils.write_list_to_file_as_string(filename=os.path.join(SAVE_DIR, 'mp_names_saved.txt'), liststrings=relevant_mp_columns+additional_feature_engineering_cols)

    shot_list_aug = os.listdir(args.raw_folder_name_aug)
    shot_list_jet = os.listdir(args.raw_folder_name_jet)
    
    shot_list_aug = [(shot, args.raw_folder_name_aug, 'AUG') for shot in shot_list_aug]
    shot_list_jet = [(shot, args.raw_folder_name_jet, 'JET') for shot in shot_list_jet]
    shot_list = shot_list_jet + shot_list_aug 

    jet_pdb = datautils.get_jetpdb(args.jetpdb_filename)

    if args.mp: 
        with Pool(psutil.cpu_count(logical=False) -2) as pool: 
            pool.map(build, shot_list)
    else: 
        for shot in shot_list: 
            build(shot)