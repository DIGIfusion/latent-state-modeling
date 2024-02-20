from data.base_torch_dataset import BaseMemMapDataset
from data.pulse_dataset import PulseDataset
from data.slice_dataset import SliceDataset
from data._utils import load_csv

import torch 
import random 
import pickle 
from typing import List 

class DatasetInterface(): 
    def __init__(self, return_type: str, **kwargs):
        self.dataset = PulseDataset(**kwargs) if return_type == 'pulse' else SliceDataset(**kwargs)
        train_idxs, val_idxs, test_idxs = self.get_train_valid_test_split_idxs(kwargs.get('split', 0.5))
        self.train_shots, self.val_shots, self.test_shots = self.get_dset_shot_nums_from_idxs(train_idxs, val_idxs, test_idxs)
        print(f'Train/val/test size {len(train_idxs)}/{len(val_idxs)}/{len(test_idxs)}')
        self.train_dataset = torch.utils.data.Subset(self.dataset, train_idxs)
        self.valid_dataset = torch.utils.data.Subset(self.dataset, val_idxs)
        self.test_dataset = torch.utils.data.Subset(self.dataset, test_idxs)
        print(f'Dataset length : {len(self.dataset)}')
        self.filter_mps_conditional = kwargs.get('filter_mps_conditional')
        self.filter_mps = kwargs.get('filter_mps')
        
    def get_train_valid_test_split_idxs(self, split): 
        if isinstance(split, float): 
            all_idxs = list(range(len(self.dataset)))
            random.shuffle(all_idxs)
            train_split, val_split = split, 0.2
            train_idxs = all_idxs[:int(len(all_idxs)*train_split)]
            val_test_idxs = all_idxs[int(len(all_idxs)*train_split):]
            val_idxs = val_test_idxs[:int(len(val_test_idxs)*val_split)]
            test_idxs = val_test_idxs[int(len(val_test_idxs)*val_split):]
        elif isinstance(split, dict): 
            # train_idxs, val_idxs, test_idxs = self.split['train_idxs'], self.split['valid_idxs'], self.split['test_idxs']  
            train_shots, val_shots, test_shots = self.split['train_shots'], self.split['test_shots'], self.split['test_shots']
        elif isinstance(split, str): 
            shot_numbers = self.dataset.shot_numbers
            if split == 'iter_baseline': 
                queries = [31147, 31148, 32472, 34454]
                train_idxs, val_idxs, test_idxs = [[shot_numbers.index(q) for q in queries] for i in [0, 1, 2]]
            elif split == 'iter_baseline_low_nu':
                # 35537, 35538, 35539, (Disr, 35540, 35536)
                queries = [35536, 35537,35538, 35539]
                train_idxs, val_idxs, test_idxs = [[shot_numbers.index(q) for q in queries] for i in [0, 1, 2]]
            elif split == 'program': 
                extract_shot_nums = lambda groups: [shot_num for group in groups for shot_num in group[1].index]
                df = load_csv('dataset_metadata.csv', self.dataset.data_path)
                grouped = list(df.groupby('program'))
                random.shuffle(grouped)
                train_split, val_split = 0.6, 0.2
                train_groups = grouped[:int(len(grouped)*train_split)]
                val_test_groups = grouped[int(len(grouped)*train_split):]
                val_groups = val_test_groups[:int(len(val_test_groups)*val_split)]
                test_groups = val_test_groups[int(len(val_test_groups)*val_split):]

                train_idxs = extract_shot_nums(train_groups) 
                val_idxs =  extract_shot_nums(val_groups)
                test_idxs =  extract_shot_nums(test_groups)
        else: 
            raise ValueError('Split variable not configured propoerly, should be float, str, or dict')
        return train_idxs, val_idxs, test_idxs
    
    def log_training_discharges(self, fname: str): 
        shot_dict = {'train_shots': self.train_shots, 'val_shots': self.val_shots, 'test_shots': self.test_shots}
        with open(fname, 'wb') as file: 
            pickle.dump(shot_dict, file)
        

    def get_dset_shot_nums_from_idxs(self, train_idxs: List[int], val_idxs: List[int], test_idxs: List[int]): 
        shot_numbers = self.dataset.shot_numbers
        train_shots, val_shots, test_shots = [[shot_numbers[q] for q in shot_idxs] for shot_idxs in [train_idxs, val_idxs, test_idxs]]
        return train_shots, val_shots, test_shots

    @property 
    def train(self): 
        return self.train_dataset
    @property 
    def valid(self): 
        return self.valid_dataset
    @property 
    def test(self): 
        return self.test_dataset