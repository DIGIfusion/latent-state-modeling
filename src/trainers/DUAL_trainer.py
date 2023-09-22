from .basetrainer import TrainInterface
from common.interfaces import M, D
import torch 
from torch import distributions as dist
from torch.nn import functional as F

class DualTrainInterface(TrainInterface): 
    model_interface = [M.vae]
    data_interface = [D.dynamic]
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_rollout = kwargs.get('num_rollout', 0)
        self.rollout_increase_step_amount = kwargs.get('rollout_increase_step_amount', 0)
        self.rollout_start = kwargs.get('rollout_start', 10000)
        self.rollout_step_interval = kwargs.get('rollout_step_interval', 10000)
        self.loss_hyperparameters = kwargs.get('loss_hyperparams', {
                        "beta_kld_transition": 1.0,
                        "beta_observations": 1.0,
                        "beta_kld_posterior_normal": 1.0, 
                        "beta_actions_cond": 1.0},)

    def train_step(self, batch, epoch, batch_idx, loader, **kwargs): 
        batch = tuple(item.transpose(0, 1) for item in batch)
        device = self.device

        obs_interface: M = self.model.observational.model_interface
        trans_interface: M = self.model.transitional.model_interface

        loss = torch.tensor([0.0], device=device)
        loss_dict = {}
        if obs_interface in [M.vae]: 
            observations, actions, radii, times = batch 
        elif obs_interface in [M.vae_aux]: 
            observations, actions, radii, times, actions_conditional = batch 

        batch_terminus = times > 0

        if self.num_rollout > 0: 
            num_rollout = torch.randint(0, self.num_rollout+1, (1,)).item() if self.num_rollout != 1 else 1
        else: 
            num_rollout = 0

        """ Observational Model """
        if obs_interface in [M.vae]: 
            observations_out = self.model.forward_observational(observations, actions)
        elif obs_interface in [M.vae_aux]: 
            observations_out = self.model.forward_observational(observations, actions_conditional)
        else: 
            raise NotImplementedError('Observational model not supported yet for SRL modeling!')
        if obs_interface in [M.vae]: 
            observations_recons, [posterior_states, posterior_state_locs, posterior_state_scales] = observations_out 
        elif obs_interface in [M.vae_aux]: 
            observations_recons, [posterior_states, posterior_state_locs, posterior_state_scales], action_reconstructions = observations_out 
        
        
        obs_model_observation_recon_loss = F.mse_loss(observations_recons, observations, reduction='none').sum(dim=(2, 3))[batch_terminus].mean()
        loss += self.loss_hyperparameters['beta_observations']*obs_model_observation_recon_loss
        loss_dict['recon_obs'] = obs_model_observation_recon_loss
        if obs_interface in [M.vae, M.vae_aux]: 
            obs_model_kl_loss = dist.kl_divergence(dist.Normal(posterior_state_locs, posterior_state_scales), dist.Normal(0, 1)).sum(dim=-1)[batch_terminus].mean()
            loss_dict['kl_obs'] = obs_model_kl_loss
            loss += self.loss_hyperparameters['beta_kld_posterior_normal']*obs_model_kl_loss
        if obs_interface in [M.vae_aux]: 
            actions_cond_loss = F.mse_loss(action_reconstructions, actions_conditional, reduction='none').sum(dim=(-1))[batch_terminus].mean()
            loss_dict['actions_cond'] = actions_cond_loss
            loss += self.loss_hyperparameters['beta_actions_cond']*actions_cond_loss
        """ Forward Model """

        # TODO: More transition models :D 
        if num_rollout != 0: 
            transition_loss_items = self.model.forward_transitional(posterior_states, posterior_state_locs, posterior_state_scales, actions[:-1], batch_terminus, num_rollout=num_rollout)
            _, prior_locs, prior_scales, _, _posterior_locs, _posterior_scales, padding_states = transition_loss_items

            trans_model_kl_loss = dist.kl_divergence(dist.Normal(_posterior_locs, _posterior_scales), dist.Normal(prior_locs, prior_scales)).sum(dim=2)[padding_states].mean()
        else: 
            trans_model_kl_loss = torch.zeros_like(loss_dict['recon_obs'], device=device)
        loss += self.loss_hyperparameters['beta_kld_transition']*trans_model_kl_loss
        loss_dict['kl_trans'] = trans_model_kl_loss

        if (epoch + 1) == self.rollout_start and (batch_idx == len(loader) - 1): 
            self.num_rollout = 1
            print(f'Starting rollout next epoch {epoch}!')
            
        elif (epoch + 2) > self.rollout_start and (epoch + 1) % self.rollout_step_interval == 0  and (batch_idx == len(loader) - 1):
            self.num_rollout += self.rollout_increase_step_amount 
            print(f'Adding to rollout: rollout = {self.num_rollout}')
       
        return (loss, loss_dict, [observations_recons, None])