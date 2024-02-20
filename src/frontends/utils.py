from typing import Union
import os 
from configs.parsing import get_config_file, set_model_config_based_on_dataset
import argparse 

import data 
import models 
import torch 

from common.interfaces import D, M


def load_model(run_name: str, model_name: Union[str, int], model_save_dir='/scratch/project_2005083/latent-state-modeling/src/saved_models'):
    config_name = 'SRL_1'
    model_path = os.path.join(model_save_dir, f'{run_name}/{model_name}.pth')
    config_filename = os.path.join(model_save_dir, f'{run_name}/{config_name}.json')
    print(f'Loading Model from: {model_path}')
    print(f'Model Config File found at {config_filename}')
    config_dict = get_config_file(config_filename)
    d = {**config_dict}
    args = argparse.Namespace(**d)


    data_interface = data.DatasetInterface(args.data.pop('return_type'), **args.data)
    args = set_model_config_based_on_dataset(args, data_interface)
    model_interface = getattr(models, args.model.pop('object'))(**args.model)

    if isinstance(model_interface, torch.nn.Module): 
        state_dict = torch.jit.load(model_path, map_location=torch.device('cpu'))
        model_interface.load_state_dict(state_dict.state_dict())
        print('Loaded Module from state_dict')
    elif isinstance(model_interface, torch.jit.TorchScript): 
        import io 
        with open(model_path, 'rb') as f:
            buffer = io.BytesIO(f.read())
        buffer.seek(0)
        loaded_module = torch.jit.load(buffer, map_location=torch.device('cpu'))
        model_interface.load_state_dict(loaded_module.state_dict())
        print('Loaded Module from jit')

    model_interface = model_interface.eval()

    return model_interface, data_interface

def uncertainty_quantification_1(queried_shot, model_interface, data_interface): 
    m_interface: M = model_interface.observational.model_interface
    d_interface: D = model_interface.data_interface
    # Should return all of the plots for a single shot
    batch_on_device = get_batch_on_device(queried_shot, data_interface)
    if m_interface in [M.DIVA, M.vae]: 
        observations, actions, radii, time = batch_on_device
        actions_cond, actions_cond_recon = [], []
    elif m_interface in  [None]: 
        observations, actions, radii, time, actions_cond = batch_on_device 
    batch_terminus = time > 0

    with torch.no_grad():
        observations_out = model_interface.forward_observational(observations, actions)

    if m_interface == M.DIVA: 
        [observations_recons, action_recons, [state_obs, state_mu_obs, state_std_devs_obs]], [observation_reconstructions_prior, action_reconstructions_prior, [prior_state, prior_state_locs, prior_state_scales]] = observations_out # model_interface.inference_all(observations, actions)
    elif m_interface == M.vae: 
        observations_recons, [state_obs, state_mu_obs, state_std_devs_obs] = observations_out

    T = batch_terminus.sum() -1 # actions.size(0) -1 
    states, states_mu, states_std_devs = state_obs[0], state_mu_obs[0], state_std_devs_obs[0]

    N_SAMPLES = 1000
    initial_samples = torch.distributions.Normal(states_mu, states_std_devs).sample((N_SAMPLES,)).squeeze(1)

    samples_rollout, samples_mean_rollout = initial_samples, initial_samples
    predicted_states_sampled_initial_mean_rollout = [initial_samples]
    predicted_states_sampled_initial_sampled_rollout = [initial_samples]


    with torch.no_grad(): 
        for t in range(T): 
            act = actions[t].repeat(N_SAMPLES, 1)
            sampled_state_actions = torch.cat([samples_rollout, act], dim=-1)
            sampled_state_actions_mean = torch.cat([samples_mean_rollout, act], dim=-1)

            _, _mu, _std_dev = model_interface.transitional.forward_single(sampled_state_actions)
            samples_rollout = torch.distributions.Normal(_mu, _std_dev).sample()

            _, _mu_mean, _std_dev_mean = model_interface.transitional.forward_single(sampled_state_actions_mean)
            samples_mean_rollout = _mu_mean

            predicted_states_sampled_initial_mean_rollout.append(samples_mean_rollout)
            predicted_states_sampled_initial_sampled_rollout.append(samples_rollout)
            
        predicted_states_sampled_initial_mean_rollout = torch.stack(predicted_states_sampled_initial_mean_rollout, dim=0)
        predicted_states_sampled_initial_sampled_rollout = torch.stack(predicted_states_sampled_initial_sampled_rollout, dim=0)
    
        predicted_observations_sampled_initial_mean_rollout = model_interface.infer_from_state(predicted_states_sampled_initial_mean_rollout)
        predicted_observations_sampled_initial_sampled_rollout = model_interface.infer_from_state(predicted_states_sampled_initial_sampled_rollout)
    
    predicted_observations_sampled_initial_mean_rollout = data_interface.dataset.denorm_profs(predicted_observations_sampled_initial_mean_rollout)
    predicted_observations_sampled_initial_sampled_rollout = data_interface.dataset.denorm_profs(predicted_observations_sampled_initial_sampled_rollout)
    _observations_recons = data_interface.dataset.denorm_profs(observations_recons)[:batch_terminus.sum()]
    _observations = data_interface.dataset.denorm_profs(observations)[:batch_terminus.sum()]
    data_x = radii[:batch_terminus.sum()].squeeze()[0]
    data_time = time[:batch_terminus.sum()].squeeze()
    return predicted_observations_sampled_initial_mean_rollout, predicted_observations_sampled_initial_sampled_rollout, _observations, _observations_recons,  data_x, data_time


def get_batch_on_device(queried_shot: Union[str, int], data_interface): 
    shot_numbers = data_interface.dataset.shot_numbers
    rel_idx = shot_numbers.index(int(queried_shot))
    *batch, shot_num = data_interface.dataset.__getitem__(rel_idx)
    batch_on_device = tuple(item.float().unsqueeze(1) for item in batch)
    return batch_on_device

def shot_prediction(queried_shot: int, model_interface, data_interface):
    m_interface: M = model_interface.observational.model_interface
    d_interface: D = model_interface.data_interface
    # Should return all of the plots for a single shot
    batch_on_device = get_batch_on_device(queried_shot, data_interface)
    if m_interface in [M.DIVA, M.vae]: 
        observations, actions, radii, time = batch_on_device
        actions_cond, actions_cond_recon = [], []
    elif m_interface in  [None]: 
        observations, actions, radii, time, actions_cond = batch_on_device 
    batch_terminus = time > 0

    
    with torch.no_grad():
        observations_out = model_interface.forward_observational(observations, actions)
    
    if m_interface == M.DIVA: 
        # observation_reconstructions, [posterior_state, posterior_state_locs, posterior_state_scales], [posterior_state_c, posterior_state_locs_c, posterior_state_scales_c], [posterior_state_not_c, posterior_state_locs_not_c, posterior_state_scales_not_c], [prior_state, prior_state_locs, prior_state_scales], action_reconstructions = out
        [observation_reconstructions_encoder, action_reconstructions_encoder, [posterior_state, posterior_state_locs, posterior_state_scales]], [observation_reconstructions_prior, action_reconstructions_prior, [prior_state, prior_state_locs, prior_state_scales]] = observations_out # model_interface.inference_all(observations, actions)
    elif m_interface == M.vae: 
        observation_reconstructions_encoder, [posterior_state, posterior_state_locs, posterior_state_scales] = observations_out


    with torch.no_grad(): 
        observations_trans, state_trans, [state_mu_trans, state_std_devs_trans] = model_interface.pulse_inference([posterior_state, posterior_state_locs, posterior_state_scales], actions[:-1])
        prior_state, prior_state_locs, prior_state_scales = torch.clone(posterior_state), torch.clone(posterior_state_locs), torch.clone(posterior_state_scales)
        prior_state[1:] =  state_trans
        prior_state_locs[1:] =  state_mu_trans
        prior_state_scales[1:] =  state_std_devs_trans

        observation_reconstructions_trans = torch.clone(observation_reconstructions_encoder)
        observation_reconstructions_trans[1:] = observations_trans

    time = time[batch_terminus].numpy()
    data_x = radii[0].cpu().numpy()[0]
    if m_interface == M.DIVA: 
        latent_space_to_return = [[posterior_state, posterior_state_locs, posterior_state_scales], [prior_state, prior_state_locs, prior_state_scales]] 
        observations_to_return = [data_interface.dataset.denorm_profs(obs)[batch_terminus] for obs in [observations, observation_reconstructions_encoder, observation_reconstructions_prior]]
        actions_to_return = [data_interface.dataset.denorm_mps(acts) for acts in [actions, action_reconstructions_encoder, action_reconstructions_prior]]
    elif m_interface == M.vae: 
        latent_space_to_return = [ [arr[batch_terminus].squeeze() for arr in group] for group in [[posterior_state, posterior_state_locs, posterior_state_scales], [prior_state, prior_state_locs, prior_state_scales]]]
        observations_to_return = [data_interface.dataset.denorm_profs(obs)[batch_terminus] for obs in [observations, observation_reconstructions_encoder, observation_reconstructions_trans]]
        actions_to_return = [data_interface.dataset.denorm_mps(acts)[batch_terminus] for acts in [actions]]
    else: 
        latent_space_to_return, observations_to_return, actions_to_return = [], [], []
    
    return observations_to_return, actions_to_return, time, data_x, batch_terminus, latent_space_to_return
