from .base_torch_dataset import BaseMemMapDataset
import torch
import torch.nn.functional as F
from typing import List, Tuple, Callable
import os 
from common.interfaces import D
class SliceDataset(BaseMemMapDataset): 
    return_type = D.slice
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_action_index_filtering(self, actions_to_filter: List[str] | None = None) -> Tuple[List[int], List[str], Callable]: 
        if actions_to_filter is None: 
            filter_mps_idxs, filter_mps_names, filter_mps_transform = None, [None], lambda u: u 
        else: 
            filter_mps_idxs = [idx for idx in range(len(self.saved_mp_names)) if self.saved_mp_names[idx] in actions_to_filter]
            filter_mps_names = [self.saved_mp_names[i] for i in filter_mps_idxs]
            print(f"Filtering MPs, keeping {filter_mps_names}")
            # filter_mps_transform = lambda u: u[:, filter_mps_idxs]
            filter_mps_transform = lambda u: u[filter_mps_idxs]
        return filter_mps_idxs, filter_mps_names, filter_mps_transform
    
    def get_pulse_and_slice_information(self, ): 
        total_num_pulses: int = len(self.data['mps'])
        list_num_slices: torch.Tensor = torch.tensor([len(self.data['mps'][i]) for i in range(len(self.data['mps']))])
        list_sampled_indicies: List[torch.Tensor] | None = None 
        cumsum_num_slices: torch.Tensor = torch.cumsum(list_num_slices, dim=0) - 1
        total_num_slices = int(sum(list_num_slices))
        shot_numbers: List[int] = sorted([int(fname.split('_')[0]) for fname in os.listdir(self.data_path) if fname.endswith('TIME.npy')])

        return shot_numbers, total_num_pulses, total_num_slices, list_num_slices, list_sampled_indicies, cumsum_num_slices
    
    def get_pulse_idx(self, idx: int) -> Tuple[int, int]:
        """A fun function that calculates which slice to take from the mmap
        Since the mmap is (imo) a list of pulses, we need to find which pulse the queried idx is coming from. 
        This is calculated by looking at the minimum value of the cumulative sum of all the slices across pulses that are >= idx 
        Then the internal pulse slice idx is just queried idx subtracted by the cumulative sum up to that pulse ( + 1)
        """
        pulse_idx: int = torch.where(self.cumsum_num_slices >= idx)[0][0].item()
        slice_idx: int = int(self.cumsum_num_slices[pulse_idx] - idx)
        return pulse_idx, slice_idx
    def __len__(self): 
        return self.total_num_slices
    
    def __getitem__(self, idx): 
        pulse_idx_to_take_from, slice_idx_to_take_from = self.get_pulse_idx(idx) 
        sample_pulse_profs = self.data['profs'][pulse_idx_to_take_from][slice_idx_to_take_from]
        sample_pulse_mps = self.data['mps'][pulse_idx_to_take_from][slice_idx_to_take_from]
        sample_pulse_time = self.data['time'][pulse_idx_to_take_from][slice_idx_to_take_from]
        sample_pulse_radii = self.data['radii'][pulse_idx_to_take_from][slice_idx_to_take_from]
        shot_num = self.shot_numbers[pulse_idx_to_take_from]
        if 'P_TOT/P_LH' in self.filter_mps_names: # TODO: this has to be moved in the future
            d_tot_idx = self.saved_mp_names.index('P_TOT/P_LH')
            sample_pulse_mps[d_tot_idx] = torch.clamp(sample_pulse_mps[d_tot_idx], min=1e-5, max=20.0)
        
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
