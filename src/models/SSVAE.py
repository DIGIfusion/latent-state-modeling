from models.base import DualModelBaseInterface, ModelInterface
from common.interfaces import M, D 
# from .model_utils import bottle, pseudo_bottle
from . import networks, model_utils
# from .networks import Encoder_1D, Decoder_1D, Regressor, Prior
import torch
from torch.nn import functional as F
import torch.distributions as dist

"""
This is in a way a combination of DIVA and CCVAE 

You have the following latent subspace 
- (disentangled) Conditional Class variables, which use a single dimension of latent space to predict a given quantity 
- (entangled) Class variables, remaining class variables which are entangled across multiple latent dims
- Remaining stochastic which contain extra info not related to class variables   
"""

class SSVAE(ModelInterface): 
    model_interface = M.ssvae
    data_interface = D.dynamic
    def __init__(self, state_size_c: int, state_size_not_c: int, state_size_stochastic: int, observation_dimensionality, action_dimensionality, action_dimensionality_conditional, cond_prior_conditional_args: dict, cond_prior_args: dict, aux_reg_conditional_args: dict, aux_reg_args: dict, encoder_args: dict, decoder_args: dict, return_type: str = 'pulse', min_std_dev = 0.01, **kwargs) -> None:
        self.data_interface = D.dynamic if return_type == 'pulse' else D.slice
        super(SSVAE, self).__init__()
        state_size_c, state_size_not_c, state_size_stochastic = model_utils.parse_cond_prior_args_for_latent_dims(self.model_interface, cond_priors=[cond_prior_conditional_args, cond_prior_args], aux_regs=[aux_reg_conditional_args, aux_reg_args], original_latent_spaces_sizes=[state_size_c, state_size_not_c, state_size_stochastic], action_dimensionalities=[action_dimensionality_conditional, action_dimensionality])
        self.min_std_dev = min_std_dev
        self.state_size_conditional = state_size_c
        self.state_size_non_conditional = state_size_not_c
        self.state_size_stoch = state_size_stochastic
        self.state_size = state_size_c + state_size_not_c + state_size_stochastic 

        self.condprior_conditional_class = getattr(networks, cond_prior_conditional_args.pop("object"))(in_dim=action_dimensionality_conditional, out_dim=state_size_c, **cond_prior_conditional_args) 
        self.condprior_class = getattr(networks, cond_prior_args.pop("object"))(in_dim=action_dimensionality, out_dim=state_size_not_c, **cond_prior_args) 
        self.encoder = networks.Encoder_1D(self.state_size, input_dimensionality=observation_dimensionality, **encoder_args, )
        self.decoder = networks.Decoder_1D(self.state_size, input_dimensionality=observation_dimensionality, **decoder_args, )
        self.auxreg_conditional_class = getattr(networks, aux_reg_conditional_args.pop("object"))(in_dim=state_size_c, out_dim=action_dimensionality_conditional, **aux_reg_conditional_args)
        self.auxreg_class = getattr(networks, aux_reg_args.pop("object"))(in_dim=state_size_not_c, out_dim=action_dimensionality, **aux_reg_args)
        self.bottler = model_utils.bottle if self.data_interface == D.dynamic else model_utils.pseudo_bottle 

    def conditional_inference(self, action: torch.Tensor, from_mean: bool = True): 
        pass 
    def infer_from_state(self, state:torch.Tensor):
        pass 
    def forward(self, observation: torch.Tensor, actions_non_conditional:torch.Tensor, actions_conditional: torch.Tensor): 

        posterior_state_locs, posterior_state_scales = self.bottler(self.encoder, (observation, ))
        posterior_state_scales = F.softplus(posterior_state_scales) + self.min_std_dev
        posterior_state = posterior_state_locs + posterior_state_scales*torch.rand_like(posterior_state_locs)

        posterior_state_locs_c, posterior_state_locs_not_c, posterior_state_locs_stoch = posterior_state_locs.split([self.state_size_conditional, self.state_size_non_conditional, self.state_size_stoch], -1)
        posterior_state_scales_c, posterior_state_scales_not_c, posterior_state_scales_stoch = posterior_state_scales.split([self.state_size_conditional, self.state_size_non_conditional, self.state_size_stoch], -1)
        posterior_state_c, posterior_state_not_c, posterior_state_stoch = posterior_state.split([self.state_size_conditional, self.state_size_non_conditional, self.state_size_stoch], -1)

        observation_reconstructions = self.bottler(self.decoder,( posterior_state, ))
        action_conditional_reconstructions = self.auxreg_conditional_class(posterior_state_c)
        action_nonconditional_reconstructions = self.auxreg_class(posterior_state_not_c)

        prior_state_conditional_locs, prior_state_conditional_scales = self.condprior_conditional_class(actions_conditional)
        prior_state_conditional_scales = F.softplus(prior_state_conditional_scales) + self.min_std_dev
        prior_state_conditional = prior_state_conditional_locs + prior_state_conditional_scales*torch.rand_like(prior_state_conditional_locs)

        prior_state_non_conditional_locs, prior_state_non_conditional_scales = self.condprior_class(actions_non_conditional)
        prior_state_non_conditional_scales = F.softplus(prior_state_non_conditional_scales) + self.min_std_dev
        prior_state_non_conditional = prior_state_non_conditional_locs + prior_state_non_conditional_scales*torch.rand_like(prior_state_non_conditional_locs)

        return observation_reconstructions, [posterior_state, posterior_state_locs, posterior_state_scales], [posterior_state_c, posterior_state_locs_c, posterior_state_scales_c], [posterior_state_not_c, posterior_state_locs_not_c, posterior_state_scales_not_c],[posterior_state_stoch, posterior_state_locs_stoch, posterior_state_scales_stoch], [prior_state_conditional, prior_state_conditional_locs, prior_state_conditional_scales], [prior_state_non_conditional, prior_state_non_conditional_locs, prior_state_non_conditional_scales], [action_nonconditional_reconstructions, action_conditional_reconstructions]
