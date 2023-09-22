from .basetrainer import TrainInterface
from common.interfaces import M, D
import torch 
from torch import distributions as dist
from torch.nn import functional as F

class DIVATrainInterface(TrainInterface): 
    model_interface = [M.DIVA]
    data_interface = [D.dynamic, D.slice]
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_hyperparameters: dict = kwargs.get('loss_hyperparams', {
                "beta_kld_prior_posterior": 1.,
                "beta_observations": 1.,
                "beta_actions": 1., 
                "beta_kld_condprior_posterior": 1., 
                "beta_kld_class_prior": 0., 
                "beta_kld_notclass_prior": 0., 
                },)
        self.loss_fn = getattr(F, kwargs.get('loss_fn', 'mse_loss'))
        
    def train_step(self, batch, epoch, batch_idx, loader, **kwargs): 
        data_interface = self.data.dataset.return_type
        model_interface = self.model.model_interface

        if  data_interface == D.dynamic: 
            batch = tuple(item.transpose(0, 1) for item in batch)
            obs_sum_dims = (2, 3)
        else: 
            obs_sum_dims = (1, 2)
        device = self.device

        loss = torch.tensor([0.0], device=device)
        loss_dict = {}

        if model_interface in [M.DIVA, M.vae]: 
            observations, actions, radii, times = batch 
        elif model_interface in [M.ssvae]:
            observations, actions, radii, times, actions_conditional = batch 

        batch_terminus = times > 0 if data_interface == D.dynamic else [True]*len(times)
        if model_interface in [M.DIVA]: 
            observation_reconstructions, [posterior_state, posterior_state_locs, posterior_state_scales], [posterior_state_c, posterior_state_locs_c, posterior_state_scales_c], [posterior_state_not_c, posterior_state_locs_not_c, posterior_state_scales_not_c],[prior_state, prior_state_locs, prior_state_scales], action_reconstructions = self.model.forward(observations, actions)
        elif model_interface in [M.ssvae]: 
            observation_reconstructions, [posterior_state, posterior_state_locs, posterior_state_scales], [posterior_state_c, posterior_state_locs_c, posterior_state_scales_c], [posterior_state_not_c, posterior_state_locs_not_c, posterior_state_scales_not_c],[posterior_state_stoch, posterior_state_locs_stoch, posterior_state_scales_stoch], [prior_state_conditional, prior_state_conditional_locs, prior_state_conditional_scales], [prior_state_non_conditional, prior_state_non_conditional_locs, prior_state_non_conditional_scales], [action_nonconditional_reconstructions, action_conditional_reconstructions] = self.model.forward(observations, actions, actions_conditional)
        elif model_interface in [M.vae]: 
            observation_reconstructions, [posterior_state, posterior_state_locs, posterior_state_scales] = self.model.forward(observations, actions)

        observation_recon_loss = self.loss_fn(observations, observation_reconstructions, reduction='none').sum(dim=obs_sum_dims) 
        
        if model_interface in [M.DIVA]: 
            action_recon_loss = self.loss_fn(actions, action_reconstructions, reduction='none').sum(dim=-1)
            kl_cond_prior_posterior = dist.kl_divergence(dist.Normal(posterior_state_locs_c, posterior_state_scales_c), dist.Normal(prior_state_locs, prior_state_scales)).sum(dim=-1)
            if not (self.loss_hyperparameters.get('beta_kld_class_prior', 0.0) or self.loss_hyperparameters.get('beta_kld_notclass_prior', 0.0)):  
                kl_prior_posterior = self.loss_hyperparameters["beta_kld_prior_posterior"]*dist.kl_divergence(dist.Normal(posterior_state_locs, posterior_state_scales), dist.Normal(0, 1)).sum(dim=-1)
            else: 
                kl_prior_posterior_class = dist.kl_divergence(dist.Normal(posterior_state_locs_c, posterior_state_scales_c), dist.Normal(0, 1)).sum(dim=-1)
                kl_prior_posterior_not_class = dist.kl_divergence(dist.Normal(posterior_state_locs_not_c, posterior_state_scales_not_c), dist.Normal(0, 1)).sum(dim=-1)
                kl_prior_posterior =  self.loss_hyperparameters["beta_kld_notclass_prior"]*kl_prior_posterior_not_class + self.loss_hyperparameters["beta_kld_class_prior"]*kl_prior_posterior_class
                loss_dict['kl_prior_post_class']  = kl_prior_posterior_class[batch_terminus].mean()
                loss_dict['kl_prior_post_notclass']  = kl_prior_posterior_not_class[batch_terminus].mean()
        elif model_interface in [M.ssvae]: 
            action_recon_loss = self.loss_fn(actions, action_nonconditional_reconstructions, reduction='none').sum(dim=-1)
            action_recon_loss += self.loss_fn(actions_conditional, action_conditional_reconstructions, reduction='none').sum(dim=-1)
            action_reconstructions = [action_nonconditional_reconstructions, action_conditional_reconstructions]

            kl_cond_prior_posterior_conditional =  dist.kl_divergence(dist.Normal(posterior_state_locs_c, posterior_state_scales_c), dist.Normal(prior_state_conditional_locs, prior_state_conditional_scales)).sum(dim=-1)
            kl_cond_prior_posterior_non_conditional =  dist.kl_divergence(dist.Normal(posterior_state_locs_not_c, posterior_state_scales_not_c), dist.Normal(prior_state_non_conditional_locs, prior_state_non_conditional_scales)).sum(dim=-1)
            kl_cond_prior_posterior = self.loss_hyperparameters.get('beta_kl_condprior_posterior_conditional', 1.0)*kl_cond_prior_posterior_non_conditional + self.loss_hyperparameters.get('beta_kl_condprior_posterior_conditional', 1.0)*kl_cond_prior_posterior_conditional
            for _name, _loss in zip(['conditional', 'non_conditional'], [kl_cond_prior_posterior_conditional, kl_cond_prior_posterior_non_conditional]): 
                loss_dict[f'kl_condprior_{_name}'] = _loss[batch_terminus].mean()

            kl_prior_posterior_conditional = dist.kl_divergence(dist.Normal(posterior_state_locs_c, posterior_state_scales_c), dist.Normal(0, 1)).sum(dim=-1)
            kl_prior_posterior_non_conditional = dist.kl_divergence(dist.Normal(posterior_state_locs_not_c, posterior_state_scales_not_c), dist.Normal(0, 1)).sum(dim=-1)
            kl_prior_posterior_stoch = dist.kl_divergence(dist.Normal(posterior_state_locs_stoch, posterior_state_scales_stoch), dist.Normal(0, 1)).sum(dim=-1)
            kl_prior_posterior = self.loss_hyperparameters.get('beta_kl_prior_posterior_stoch', 1.0)*kl_prior_posterior_stoch + self.loss_hyperparameters.get('beta_kl_prior_posterior_non_conditional', 1.0)*kl_prior_posterior_non_conditional + self.loss_hyperparameters.get('beta_kl_prior_posterior_conditional', 1.0)*kl_prior_posterior_conditional
            
            for _name, _loss in zip(['conditional', 'non_conditional', 'stoch'], [kl_prior_posterior_conditional, kl_prior_posterior_non_conditional, kl_prior_posterior_stoch]): 
                loss_dict[f'kl_prior_{_name}'] = _loss[batch_terminus].mean()

        elif model_interface in [M.vae]: 
            action_reconstructions = None
            action_recon_loss = torch.zeros_like(observation_recon_loss)
            kl_cond_prior_posterior = torch.zeros_like(observation_recon_loss)
            kl_prior_posterior = dist.kl_divergence(dist.Normal(posterior_state_locs, posterior_state_scales), dist.Normal(0, 1)).sum(dim=-1)

        loss += (self.loss_hyperparameters["beta_observations"]*observation_recon_loss + 
                 self.loss_hyperparameters["beta_actions"]*action_recon_loss + 
                 self.loss_hyperparameters["beta_kld_condprior_posterior"]*kl_cond_prior_posterior + 
                 kl_prior_posterior
                 )[batch_terminus].mean()
        
        loss_dict['kl_prior_post'] = kl_prior_posterior[batch_terminus].mean()
        loss_dict['kl_cond_prior_post'] = kl_cond_prior_posterior[batch_terminus].mean()
        loss_dict['observation_recon'] = observation_recon_loss[batch_terminus].mean()
        loss_dict['action_recon'] = action_recon_loss[batch_terminus].mean()

        # TODO: Physics
        return loss, loss_dict, [observation_reconstructions, action_reconstructions]
