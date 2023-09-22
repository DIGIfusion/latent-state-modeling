from typing import List, Union, Tuple
import torch 
from common.interfaces import M
def parse_convolution_argument(arg_values: Union[List[int], int], num_layers: int, arg_name: str) -> List[int]: 
    if isinstance(arg_values, int): 
        arg_values = [arg_values]*num_layers 
    elif isinstance(arg_values, list): 
        if len(arg_values) != num_layers: 
            raise TypeError(f'{arg_name} does not match number of layers proposed by channel sizes, they must be either the same length (list of ints) or a single int for all the same {arg_name}')
    return arg_values

def get_output_size_from_block(block: torch.nn.ModuleList, sample: torch.Tensor): 
    # make sure that sample is only of batch size 1
    for lay in block: 
        sample = lay(sample) 
    return sample.shape.numel

def parse_cond_prior_args_for_latent_dims(model_interface: M, cond_priors: List[dict], aux_regs: List[dict], original_latent_spaces_sizes: List[int], action_dimensionalities: List[int]): 
    new_latent_space_sizes = original_latent_spaces_sizes.copy()
    if model_interface in [M.ssvae]: 
        prior_conditional, prior_non_conditional = cond_priors 
        reg_conditional, reg_non_conditional = aux_regs
        if 'Diagonal' in prior_conditional.get('object') or 'Diagonal' in reg_conditional.get('object'): 
            new_latent_space_sizes[0] = action_dimensionalities[0]
            prior_conditional['dim'] = action_dimensionalities[0]
            reg_conditional['dim'] = action_dimensionalities[0]
            print(f'Diagnoal prior or regressor chosen for conditional vars, so the latent dims will from {original_latent_spaces_sizes[0]} ---> {new_latent_space_sizes[0]}')
        if 'Diagonal' in prior_non_conditional.get('object')  or 'Diagonal' in reg_non_conditional.get('object'): 
            new_latent_space_sizes[1] = action_dimensionalities[1]
            prior_non_conditional['dim'] = action_dimensionalities[1]
            reg_non_conditional['dim'] = action_dimensionalities[1]
            print(f'Diagnoal prior or regressor chosen for non-conditional vars, so the latent dims will from {original_latent_spaces_sizes[0]} ---> {new_latent_space_sizes[0]}')
    elif model_interface in [M.DIVA]: 
        prior_conditional = cond_priors[0]
        reg_conditional = aux_regs[0]
        if 'Diagonal' in prior_conditional.get('object') or 'Diagonal' in reg_conditional.get('object'): 
            new_latent_space_sizes[0] = action_dimensionalities[0]
            prior_conditional['dim'] = action_dimensionalities[0]
            reg_conditional['dim'] = action_dimensionalities[0]
            print(f'Diagnoal prior or regressor chosen for conditional vars, so the latent dims will from {original_latent_spaces_sizes[0]} ---> {new_latent_space_sizes[0]}')
    elif model_interface in [M.vae_aux]: 
        reg_conditional = aux_regs[0]
        if 'Diagonal' in reg_conditional.get('object'): 
            new_latent_space_sizes[0] = action_dimensionalities[0]
            reg_conditional['dim'] = action_dimensionalities[0]
            print(f'Diagnoal regressor chosen for conditional vars, so the latent dims will from {original_latent_spaces_sizes[0]} ---> {new_latent_space_sizes[0]}')
    return new_latent_space_sizes 
def pseudo_bottle(f, x_tuple: Tuple[torch.Tensor]): 
    return f(*x_tuple)

def bottle(f, x_tuple): 
    """ A function to take an entire pulse and apply a vae to it """
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    y = f(*map(lambda x: x[0].reshape(x[1][0]*x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
    if len(y) <=2: 
        outs = []
        for y_ in y: 
            y_size = y_.size()
            out_y_ = y_.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])
            outs.append(out_y_)
    else: 
        y_size = y.size()
        outs = y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])
    return outs 