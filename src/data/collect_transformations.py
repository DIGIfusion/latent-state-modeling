import numpy as np 

class RunningStats:
    def __init__(self, name):
        self.name = name
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return np.sqrt(self.variance())

    def __repr__(self):
        return f'\n####{self.name}####\nn: {self.n}, mean: {self.mean()}, var: {self.variance()}, sd: {self.standard_deviation()}'

def get_mp_names_saved_in_arrays(data_path: str) -> list:
    with open(os.path.join(SAVE_DIR, 'mp_names_saved.txt'), 'r') as f:
        all_names_str = f.read()
        relevant_mp_columns = all_names_str.split(',')
    return relevant_mp_columns


import argparse 
import os 
from tqdm import tqdm 

parser = argparse.ArgumentParser(description='Process raw data into numpy arrays')
parser.add_argument('-af', '--array_folder_name', type=str, default='/home/kitadam/ENR_Sven/test_moxie/experiments/ICDDPS_AK/local_data/', help='Folder name under which to store the raw numpy arrays. This will be found in whatever your processed dir is.')
args = parser.parse_args()
SAVE_DIR = args.array_folder_name

profiles = sorted([os.path.join(SAVE_DIR, file) for file in os.listdir(SAVE_DIR) if file.endswith('PROFS.npy')])
mps = sorted([os.path.join(SAVE_DIR, file) for file in os.listdir(SAVE_DIR) if file.endswith('MP.npy')])
print(len(profiles), len(mps))
shot_files = zip(profiles, mps)

saved_mp_names: list = get_mp_names_saved_in_arrays(SAVE_DIR)
print(saved_mp_names)
# gas_idxs = [saved_mp_names.index(name) for name in ['D_tot', 'N_tot'] if name in saved_mp_names]


prof_mean, mp_mean = RunningStats('prof'), RunningStats('mp')
for k, (prof, mp) in tqdm(enumerate(shot_files)): 
    prof, mp = np.load(prof), np.load(mp)
    prof[:, 0]*= 1e-19
    # mp = np.nan_to_num(mp)
    # if 'P_TOT/P_LH' in saved_mp_names: 
    #     mp[:, saved_mp_names.index('P_TOT/P_LH')] = np.clip(mp[:, saved_mp_names.index('P_TOT/P_LH')], a_min=0, a_max=20)
    # if "tau_tot" in saved_mp_names: 
    #     mp[:, saved_mp_names.index("tau_tot")] = np.clip(mp[:, saved_mp_names.index('tau_tot')], a_min=1e-5, a_max=3)
    # sample_pulse_mps[:, tau_tot_idx] = torch.clamp(torch.nan_to_num(sample_pulse_mps[:, tau_tot_idx]), min=1e-5, max=3)
    for p, m in zip(prof, mp): 
        prof_mean.push(p)
        mp_mean.push(m)
    if k > 1000: 
        break

prof_means_subverted = prof_mean.mean()
prof_means_subverted[0] *= 1e19

prof_stds_subverted = prof_mean.standard_deviation()
prof_stds_subverted[0]*= 1e19

mp_means = mp_mean.mean()
mp_stds= mp_mean.standard_deviation()
print('Means shapes', mp_means.shape, mp_stds.shape)
print(mp_means, mp_stds)

transformation_dict: dict = dict(mp_means=mp_means, mp_stds=mp_stds, 
                                 prof_means=prof_means_subverted, prof_stds=prof_stds_subverted)
import pickle  


with open(os.path.join(SAVE_DIR, 'transformations.pickle'), 'wb') as file: 
    pickle.dump(transformation_dict, file)

print(f'Saved transformations to {SAVE_DIR}/transformations.pickle')
