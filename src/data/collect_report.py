import argparse 
import os 
from tqdm import tqdm 
from _utils import get_local_pulse_dict, read_yaml_input_file, get_list_of_strings_from_file, get_local_pulse_arrays
import matplotlib.pyplot as plt 
from multiprocessing import Pool
import numpy as np 

parser = argparse.ArgumentParser(description='Process raw data into numpy arrays')
parser.add_argument('-rf', '--raw_folder_name', type=str, default=None)
parser.add_argument('-af', '--array_folder_name', type=str, default='/home/kitadam/ENR_Sven/test_moxie/experiments/ICDDPS_AK/local_data/', help='Folder name under which to store the raw numpy arrays. This will be found in whatever your processed dir is.')
args = parser.parse_args()

SAVE_DIR = args.array_folder_name
RAW_DIR = args.raw_folder_name

data_config = read_yaml_input_file(fname=os.path.join(SAVE_DIR, 'DATA_CONFIG.yaml'))
saved_shot_nums = [fname.split('_')[0] for fname in os.listdir(SAVE_DIR) if (('PROFS' in fname) and fname.endswith('npy'))]

report_save_dir = os.path.join(SAVE_DIR, 'reports')
if not os.path.exists(report_save_dir): 
    os.mkdir(report_save_dir)
""" 
Some stuff to test
"""

mp_column_names = get_list_of_strings_from_file(os.path.join(SAVE_DIR, 'mp_names_saved.txt'))
rhos = [0.0, 0.3, 0.5, 0.9, 1.0]
rhos_str = [str(rho).replace('.', '_') for rho in rhos]
mp_column_names.extend([f'ne_{rho}' for rho in rhos_str] + [f'te_{rho}' for rho in rhos_str])
def process_file(shotno): 
    profiles, mps, time, radii = get_local_pulse_arrays(shotno, SAVE_DIR)
    nes = [] 
    tes = []
    for rho in rhos: 
        rho_idx = np.argmin(abs(radii[0] - rho))
        nes.append(profiles[:, 0, rho_idx])
        tes.append(profiles[:, 1, rho_idx])
    
    # core_ne, core_te = profiles[:, 0, 0], profiles[:, 1, 0]
    
    prof_info = np.stack([*nes, *tes], 1)
    return mps, prof_info



def plot_histograms(data, mp_column_names):
    # Assuming data.shape[1] is the number of parameters
    num_parameters = data.shape[1]
    for i in range(num_parameters):
        save_name = os.path.join(report_save_dir, f'hist_{mp_column_names[i]}'.replace('/', '_'))
        plt.figure()
        plt.hist(data[:, i], bins=50, alpha=0.75)
        plt.title(f'{mp_column_names[i]}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.axvline(data[:, i].min(), color='black', ls='--')
        plt.axvline(data[:, i].max(), color='black', ls='--')
        plt.savefig(save_name)
        
        plt.close()
        print(f'{mp_column_names[i]}, min: {data[:, i].min():.5}, max: {data[:, i].max():.5}')

with Pool(processes=os.cpu_count()) as pool:
    # Map the process_file function to each file_path
    results = pool.map(process_file, saved_shot_nums)

mp_results = np.concatenate([result[0] for result in results], 0)
prof_results = np.concatenate([result[1] for result in results], 0)
print(mp_results.shape, prof_results.shape)
results = np.concatenate([mp_results, prof_results], 1)
plot_histograms(results, mp_column_names)
"""
for shot_idx, shot_no in enumerate(saved_shot_nums):
    # load data 
    profiles, mps, time, radii = get_local_pulse_arrays(shot_no, SAVE_DIR)

    for mp_idx, mp_name in enumerate(mp_column_names):

        pass 
    break  
"""