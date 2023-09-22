import torch 
from torch.utils.data import Dataset
from torch.nn import functional as F
from mmap_ninja.ragged import RaggedMmap
from typing import List, Tuple, Dict, Callable
import pickle 
import os 
import numpy as np
import abc 

from ._utils import get_list_of_strings_from_file
def compute_raggedmmap(data_dir: str, data_str: str="PROFS") -> RaggedMmap: 
    """ Load or create the memmory mapped arrays """
    def load_data(paths):
        for path in paths: 
            yield np.load(path)
    save_name = os.path.join(data_dir, f'{data_str}_MMAP')
    if os.path.exists(save_name): 
        ragged_mmap = RaggedMmap(save_name, wrapper_fn=torch.tensor)
        print(f'A ragged map for {data_str} exists at {save_name} with length: {len(ragged_mmap)}')
    else: 
        relevant_paths = sorted([os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith(f"_{data_str}.npy")])
        # NB: Because of the sorted, the pulses are accessible in the memmap in the same order as the above 
        
        ragged_mmap = RaggedMmap.from_generator(out_dir=save_name, 
                                             sample_generator=load_data(relevant_paths),
                                             verbose=True, 
                                             batch_size=256, wrapper_fn=torch.tensor)
    return ragged_mmap

class BaseMemMapDataset(Dataset, metaclass=abc.ABCMeta): 
    def __init__(self, **kwargs): 
        self.data_path = kwargs.get('data_path', '/')
        self.device = kwargs.get('device', 'cpu')
        print('data path', self.data_path)
        self.data = self.get_data_from_raggedmmaps(self.data_path)
        self.shot_numbers, self.total_num_pulses, self.total_num_slices, self.list_num_slices, self.list_sampled_indicies, self.cumsum_num_slices = self.get_pulse_and_slice_information()
        self.observational_channels, self.observational_spatial_dim = self.data['profs'][0].shape[1:]
        self.saved_mp_names: list = get_list_of_strings_from_file(os.path.join(self.data_path, 'mp_names_saved.txt'))
        
        filter_mps, filter_mps_conditional = kwargs.get('filter_mps', None), kwargs.get('filter_mps_conditional', None)
        self.filter_mps_idxs, self.filter_mps_names, self.filter_mps_transform = self.get_action_index_filtering(filter_mps)
        self.filter_mps_idxs_conditional, self.filter_mps_conditional_names, self.filter_mps_conditional_transform = self.get_action_index_filtering(filter_mps_conditional)

        transformation_file = kwargs.get('transformation_file', 'transformations.pickle')
        self.mean_actions, self.std_actions, self.mean_observations, self.std_observations = self.get_transformation_means_and_stds_from_file(transformation_file)
        
        self.transform_mps = lambda u: (u - self.mean_actions)/ self.std_actions
        self.transform_profs = lambda u: (u  - self.mean_observations) / self.std_observations
        
        if self.filter_mps_transform is not None: 
            self.denorm_mps = lambda u: (u* self.std_actions[self.filter_mps_idxs]) + self.mean_actions[self.filter_mps_idxs]
        else: 
            self.denorm_mps = lambda u: (u*self.std_actions) + self.mean_actions

        if self.filter_mps_idxs_conditional is not None: 
            self.denorm_mps_cond = lambda u: (u* self.std_actions[self.filter_mps_idxs_conditional]) + self.mean_actions[self.filter_mps_idxs_conditional]
            self.norm_mps_cond = lambda u: (u - self.mean_actions[self.filter_mps_idxs_conditional]) / self.std_actions[self.filter_mps_idxs_conditional]
            print('Creating conditional clamping vector')
            self.cond_clamp_vector = ((torch.zeros((len(self.filter_mps_idxs_conditional)), device=self.device) - self.mean_actions[self.filter_mps_idxs_conditional]) / self.std_actions[self.filter_mps_idxs_conditional]).float()# .to(device)
        else: 
            self.denorm_mps_cond = lambda u: (u*self.std_actions) + self.mean_actions
            self.norm_mps_cond = lambda u: (u - self.mean_actions) / self.std_actions
        
        if kwargs.get("clamp_observations_to_reals", False): 
            print('Creating Observational clamping vector')
            self.observations_clamp_vector = ((torch.zeros((1, self.observational_channels, self.observational_spatial_dim), device=self.device) - self.mean_observations) / self.std_observations).float()
        
        self.denorm_profs = lambda u: (u*self.std_observations.to(u.device)) + self.mean_observations.to(u.device)

    def get_data_from_raggedmmaps(self, data_path) -> Dict[str, RaggedMmap]: 
        data = {}
        data['profs'] = compute_raggedmmap(data_path, data_str='PROFS') # data['profs'][0] correspond to the profiles for the first pulse in sorted(os.listdir(data_path)) 
        data['mps'] = compute_raggedmmap(data_path, data_str='MP')    
        data['radii'] = compute_raggedmmap(data_path, data_str='RADII')
        data['time'] = compute_raggedmmap(data_path, data_str='TIME')
        return data 
    
    def get_transformation_means_and_stds_from_file(self, transform_file: str) -> List[np.ndarray]: 
        if '/' not in transform_file: 
            trans_data_path = self.data_path 
        else: 
            trans_data_path = transform_file.split('transformations.pickle')[0]
            transform_file = 'transformations.pickle'

        with open(os.path.join(trans_data_path, transform_file), 'rb') as file: 
            print(f'loading transformations from {trans_data_path}/{transform_file} ')
            transformation_dict: dict = pickle.load(file)
        return [torch.from_numpy(transformation_dict[name]).to(self.device) for name in ['mp_means', 'mp_stds', 'prof_means', 'prof_stds']]


    @abc.abstractmethod 
    def get_action_index_filtering(filter_mps: List[str]): 
        raise NotImplementedError('Returning pulse type is not done yet')
    
    @abc.abstractmethod
    def get_pulse_and_slice_information(self, ): 
        # TODO: this is specific for the class
        raise NotImplementedError('Returning pulse type is not done yet')
    
    @property 
    @abc.abstractmethod
    def return_type(self): 
        raise NotImplementedError("Data return type is not set")


    @property 
    @abc.abstractmethod
    def __len__(self): 
        raise NotImplementedError('needs to be set')
    
    @property 
    @abc.abstractmethod
    def __getitem__(self): 
        raise NotImplementedError('needs to be set')