from models.base import DualModelBaseInterface, ModelInterface
from common.interfaces import M, D 
from .model_utils import bottle, pseudo_bottle
from . import networks

import torch
from torch.nn import functional as F
import torch.distributions as dist

class DIVA(ModelInterface): 
    model_interface = M.DIVA
    data_interface = D.dynamic
    def __init__(self, state_size_c: int, state_size_not_c: int, observation_dimensionality, action_dimensionality, cond_prior_args: dict, aux_reg_args: dict, encoder_args: dict, decoder_args: dict, return_type: str = 'pulse', min_std_dev = 0.01, **kwargs) -> None:
        self.data_interface = D.dynamic if return_type == 'pulse' else D.slice
        super(DIVA, self).__init__()
        self.min_std_dev = min_std_dev
        state_size = state_size_c + state_size_not_c
        self.state_size_c, self.state_size_not_c = state_size_c, state_size_not_c
        self.encoder = networks.Encoder_1D(state_size, input_dimensionality=observation_dimensionality, **encoder_args, )
        self.decoder = networks.Decoder_1D(state_size, input_dimensionality=observation_dimensionality, **decoder_args, )
        self.cond_prior = getattr(networks, cond_prior_args.pop("object"))(in_dim=action_dimensionality, out_dim=state_size_c, **cond_prior_args) # Prior(in_dim=action_dimensionality, out_dim=state_size_c, **cond_prior_args)
        self.aux_reg = networks.Regressor(in_dim=state_size_c, out_dim=action_dimensionality, **aux_reg_args)        
        self.bottler = bottle if self.data_interface == D.dynamic else pseudo_bottle 

    def inference_all(self, observation: torch.Tensor, action: torch.Tensor, from_mean: bool=True): 
        posterior_state_locs, posterior_state_scales = self.bottler(self.encoder, (observation, ))
        posterior_state_scales = F.softplus(posterior_state_scales) + self.min_std_dev
        if from_mean: 
            posterior_state = posterior_state_locs
        else: 
            posterior_state = posterior_state_locs + posterior_state_scales*torch.rand_like(posterior_state_locs)
        print(posterior_state.shape, posterior_state.shape[:-1], posterior_state.shape[0])
        observation_reconstructions_encoder, action_reconstructions_encoder = self.infer_from_state(posterior_state)
        observation_reconstructions_prior, action_reconstructions_prior, [prior_state, prior_state_locs, prior_state_scales] = self.conditional_inference(action)

        return [observation_reconstructions_encoder, action_reconstructions_encoder, [posterior_state, posterior_state_locs, posterior_state_scales]], [observation_reconstructions_prior, action_reconstructions_prior, [prior_state, prior_state_locs, prior_state_scales]]
    def conditional_inference(self, action: torch.Tensor, from_mean: bool = True): 
        state_c_locs, state_c_scales = self.cond_prior(action)
        state_c_scales = F.softplus(state_c_scales) + self.min_std_dev

        state_not_c = dist.Normal(0, 1).sample((state_c_locs.shape[0], state_c_locs.shape[1], self.state_size_not_c))
         
        if from_mean: 
            state_c = state_c_locs 
            state = torch.cat([state_c, state_not_c], -1)
            
        else: 
            state_c = state_c_locs + state_c_scales*torch.rand_like(state_c_scales)
            state = torch.cat([state_c, state_not_c], -1)

        state_locs = torch.cat([state_c_locs, torch.zeros_like(state_not_c)], -1)
        state_scales = torch.cat([state_c_scales, torch.ones_like(state_not_c)], -1)
        print(state.shape)
        observation_reconstructions, action_reconstructions = self.infer_from_state(state)
        return observation_reconstructions, action_reconstructions, [state, state_locs, state_scales]


    def infer_from_state(self, state: torch.Tensor, ): 
        state_c, state_not_c = state.split([self.state_size_c, self.state_size_not_c], -1)
         
        observation_reconstructions = self.bottler(self.decoder,(state, ))
        action_reconstructions = self.aux_reg(state_c)
        
        return observation_reconstructions, action_reconstructions
    def forward(self, observation: torch.Tensor, action: torch.Tensor): 

        posterior_state_locs, posterior_state_scales = self.bottler(self.encoder, (observation, ))
        posterior_state_scales = F.softplus(posterior_state_scales) + self.min_std_dev
        posterior_state = posterior_state_locs + posterior_state_scales*torch.rand_like(posterior_state_locs)

        posterior_state_locs_c, posterior_state_locs_not_c = posterior_state_locs.split([self.state_size_c, self.state_size_not_c], -1)
        posterior_state_scales_c, posterior_state_scales_not_c = posterior_state_scales.split([self.state_size_c, self.state_size_not_c], -1)
        posterior_state_c, posterior_state_not_c = posterior_state.split([self.state_size_c, self.state_size_not_c], -1)
         
        observation_reconstructions = self.bottler(self.decoder,( posterior_state, ))
        action_reconstructions = self.aux_reg(posterior_state_c)
        
        prior_state_locs, prior_state_scales = self.cond_prior(action)
        prior_state_scales = F.softplus(prior_state_scales) + self.min_std_dev
        prior_state = prior_state_locs + prior_state_scales*torch.rand_like(prior_state_locs)

        return observation_reconstructions, [posterior_state, posterior_state_locs, posterior_state_scales], [posterior_state_c, posterior_state_locs_c, posterior_state_scales_c], [posterior_state_not_c, posterior_state_locs_not_c, posterior_state_scales_not_c], [prior_state, prior_state_locs, prior_state_scales], action_reconstructions