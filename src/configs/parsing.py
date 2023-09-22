import json 
import os 
import argparse 

def get_config_file(filename: str) -> dict:
    if os.path.exists(filename): 
        with open(filename, 'r') as file: 
            config_file = json.load(file)
        return config_file
    else: 
        raise FileNotFoundError(filename + 'config not found')
    

def set_model_config_based_on_dataset(args: argparse.Namespace, data_interface): 
    if args.data.get('filter_mps_conditional', False): 
        args.model['action_dimensionality_conditional'] = len(args.data.get('filter_mps_conditional'))
        if args.data.get('scale_aux_reg', False): 
            args.model['scale_aux_reg'] = data_interface.dataset.cond_clamp_vector
    if args.data.get('clamp_observations_to_reals', False): 
        if args.model['object'] == 'DualModelInterface': 
            args.model['observational_model']['decoder_args']['output_clamping'] = data_interface.dataset.observations_clamp_vector
        else: 
            args.model['decoder_args']['output_clamping'] = data_interface.dataset.observations_clamp_vector

    args.model['observation_dimensionality'] =(data_interface.dataset.observational_channels, data_interface.dataset.observational_spatial_dim)
    args.model['action_dimensionality'] = len(data_interface.dataset.filter_mps_names)
    args.model['return_type'] = data_interface.dataset.return_type
    return args