from data.base_torch_dataset import BaseMemMapDataset
from data.pulse_dataset import PulseDataset
from data.slice_dataset import SliceDataset

import torch 
import random 
class DatasetInterface(): 
    def __init__(self, return_type: str, **kwargs):
        self.dataset = PulseDataset(**kwargs) if return_type == 'pulse' else SliceDataset(**kwargs)
        train_idxs, val_idxs, test_idxs = self.get_train_valid_test_split_idxs(kwargs.get('split', 0.5))
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
            train_idxs, val_idxs, test_idxs = self.split['train_idxs'], self.split['valid_idxs'], self.split['test_idxs']  
        elif isinstance(split, str): 
            shot_numbers = self.dataset.shot_numbers
            if split == 'iter_baseline': 
                queries = [31147, 31148, 32472, 34454]
            elif split == 'iter_baseline_low_nu':
                # 35537, 35538, 35539, (Disr, 35540, 35536)
                queries = [35536, 35537,35538, 35539]
            train_idxs, val_idxs, test_idxs = [[shot_numbers.index(q) for q in queries] for i in [0, 1, 2]]
        else: 
            raise ValueError('Split variable not configured propoerly, should be float, str, or dict')
        return train_idxs, val_idxs, test_idxs
    

    @property 
    def train(self): 
        return self.train_dataset
    @property 
    def valid(self): 
        return self.valid_dataset
    @property 
    def test(self): 
        return self.test_dataset