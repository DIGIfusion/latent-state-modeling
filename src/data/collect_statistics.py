import argparse 
import os 
from tqdm import tqdm 
from _utils import get_local_pulse_dict
parser = argparse.ArgumentParser(description='Process raw data into numpy arrays')
parser.add_argument('-rf', '--raw_folder_name', type=str, default=None)
parser.add_argument('-af', '--array_folder_name', type=str, default='/home/kitadam/ENR_Sven/test_moxie/experiments/ICDDPS_AK/local_data/', help='Folder name under which to store the raw numpy arrays. This will be found in whatever your processed dir is.')
args = parser.parse_args()
SAVE_DIR = args.array_folder_name
RAW_DIR = args.raw_folder_name
saved_shot_nums = [fname.split('_')[0] for fname in os.listdir(SAVE_DIR) if 'PROFS' in fname]

import pandas as pd 

save_df = {}
for shot_num in saved_shot_nums: 
    pulse_dict = get_local_pulse_dict(shot_num, RAW_DIR)
    journal = pulse_dict['journal']
    for key in journal:
        if isinstance(journal[key], bytes):
            journal[key] = journal[key].decode('utf-8')

    program = journal['program'].split('.xml')[0]
    print(shot_num, program, journal['proposal'])
    prop_num = int(journal['proposal']) if journal['proposal'] != '' else 1 
    pulse_metadata = {'proposal': prop_num, 'program': program} 
    save_df[shot_num] = pulse_metadata


# Convert the dictionary to a DataFrame
df = pd.DataFrame.from_dict(save_df, orient='index')
# Set 'shot_num' as the index
df.index.name = 'shot_num'

# Save the DataFrame to a file, e.g., CSV
df.to_csv(os.path.join(SAVE_DIR, 'dataset_metadata.csv'))
print()
print('Total shots', len(df), 'Unique proposals', len(set(df['proposal'])), 'Unique programs', len(set(df['program'])))
print('Saved metadata to {SAVE_DIR}/dataset_metadata.csv')