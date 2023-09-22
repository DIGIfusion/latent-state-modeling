from .base_torch_dataset import BaseMemMapDataset
import torch
import torch.nn.functional as F
from typing import List, Tuple, Callable
import os 
from common.interfaces import D
class PulseDataset(BaseMemMapDataset): 
    return_type = D.dynamic
    def __init__(self, **kwargs):
        self.sample_interval = kwargs.get('sample_interval', 1)
        super().__init__(**kwargs)
        self.pad_size = int(max(self.list_num_slices).item())
        self.constant_pad_val = -1
        self.chunk_size = kwargs.get('chunk_size', 10000)
    def get_action_index_filtering(self, actions_to_filter: List[str] | None = None) -> Tuple[List[int], List[str], Callable]: 
        if actions_to_filter is None: 
            filter_mps_idxs, filter_mps_names, filter_mps_transform = None, [None], lambda u: u 
        else: 
            filter_mps_idxs = [idx for idx in range(len(self.saved_mp_names)) if self.saved_mp_names[idx] in actions_to_filter]
            filter_mps_names = [self.saved_mp_names[i] for i in filter_mps_idxs]
            print(f"Filtering MPs, keeping {filter_mps_names}")
            filter_mps_transform = lambda u: u[:, filter_mps_idxs]
            # for the slice dataset filter_mps_transform = lambda u: u[filter_mps_idxs]
        return filter_mps_idxs, filter_mps_names, filter_mps_transform
    
    def get_pulse_and_slice_information(self, ): 
        total_num_pulses: int = len(self.data['mps'])
        list_num_slices: torch.Tensor = torch.tensor([len(self.data['mps'][i]) for i in range(len(self.data['mps']))])
        if self.sample_interval > 1: # Down sample time means we have to change the number of slices per pulse
            list_sampled_indicies: List[torch.Tensor] | None = [torch.arange(0, n_slices, step=self.sample_interval) for n_slices in list_num_slices]
            list_num_slices: torch.Tensor = torch.tensor([len(sample_indicies) for sample_indicies in list_sampled_indicies])
        else: 
            list_sampled_indicies: List[torch.Tensor] | None = None 
        cumsum_num_slices: torch.Tensor = torch.cumsum(list_num_slices, dim=0) - 1
        total_num_slices = int(sum(list_num_slices))
        shot_numbers: List[int] = sorted([int(fname.split('_')[0]) for fname in os.listdir(self.data_path) if fname.endswith('TIME.npy')])

        return shot_numbers, total_num_pulses, total_num_slices, list_num_slices, list_sampled_indicies, cumsum_num_slices
    
    def __len__(self): 
        return self.total_num_pulses
    
    def __getitem__(self, idx): 
        sample_pulse_profs = self.data['profs'][idx]
        sample_pulse_mps = self.data['mps'][idx]
        sample_pulse_time = self.data['time'][idx]
        sample_pulse_radii = self.data['radii'][idx]

        if self.list_sampled_indicies is not None: 
            sample_indicies = self.list_sampled_indicies[idx]
            sample_pulse_profs = sample_pulse_profs[sample_indicies]
            sample_pulse_mps = sample_pulse_mps[sample_indicies]
            sample_pulse_time = sample_pulse_time[sample_indicies]
            sample_pulse_radii = sample_pulse_radii[sample_indicies]
        pulse_padding_length = self.pad_size - sample_pulse_profs.shape[0]
        sample_pulse_profs = F.pad(sample_pulse_profs, (0, 0, 0, 0, 0, pulse_padding_length), "constant", self.constant_pad_val)
        sample_pulse_radii = F.pad(sample_pulse_radii, (0, 0, 0, pulse_padding_length), "constant", self.constant_pad_val)
        sample_pulse_mps = F.pad(sample_pulse_mps, (0, 0, 0, pulse_padding_length), "constant", self.constant_pad_val)
        sample_pulse_time = F.pad(sample_pulse_time, (0, pulse_padding_length), 'constant', -1)
        shot_num = self.shot_numbers[idx]
        if self.chunk_size != 0:
            sample_pulse_profs, sample_pulse_mps, sample_pulse_radii, sample_pulse_time = sample_pulse_profs[:self.chunk_size], sample_pulse_mps[:self.chunk_size], sample_pulse_radii[:self.chunk_size], sample_pulse_time[:self.chunk_size]

        # below is the same for slice and pulse
        sample_pulse_profs = self.transform_profs(sample_pulse_profs)
        sample_pulse_mps = self.transform_mps(sample_pulse_mps)
        # if self.filter_mps_transform is not None: 
        sample_pulse_mps_actions = self.filter_mps_transform(sample_pulse_mps)
        if self.filter_mps_idxs_conditional is not None: 
            sample_pulse_mps_conditional = self.filter_mps_conditional_transform(sample_pulse_mps) 

            return sample_pulse_profs, sample_pulse_mps_actions, sample_pulse_radii, sample_pulse_time, sample_pulse_mps_conditional, shot_num 
        else: 
            return sample_pulse_profs, sample_pulse_mps_actions, sample_pulse_radii, sample_pulse_time, shot_num # # TODO: if filter_mps is ever none, then we are fuxked
