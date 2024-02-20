from .base import ModelInterface
from common.interfaces import M, D
from torch import nn, jit 
import torch 
import torch.nn.functional as F
from . import model_utils, networks
# from .networks import Encoder_1D, Decoder_1D, Regressor, Prior

class VAE_1D(ModelInterface, jit.ScriptModule): 
    model_interface = M.vae
    data_interface = D.dynamic
    def __init__(self, state_size: int, observation_dimensionality, encoder_args: dict, decoder_args: dict, return_type: str,min_std_dev: float = 0.01,*args, **kwargs) -> None:
        # self.data_interface = D.dynamic if return_type == 'pulse' else D.slice
        super(VAE_1D, self).__init__()
        self.min_std_dev = min_std_dev
        self.state_size = state_size
        self.encoder = networks.Encoder_1D(state_size, input_dimensionality=observation_dimensionality, **encoder_args, )
        self.decoder = networks.Decoder_1D(state_size, input_dimensionality=observation_dimensionality, **decoder_args, )
        self.bottler = model_utils.bottle if self.data_interface == D.dynamic else model_utils.pseudo_bottle 


    def decode_state(self, states: torch.Tensor): 
        return self.bottler(self.decoder,(states, ))

    def forward(self, observations: torch.Tensor, *args, **kwargs): 
        state_locs, state_scales = self.bottler(self.encoder, (observations, ))
        state_scales = F.softplus(state_scales) + self.min_std_dev
        state = state_locs + state_scales*torch.rand_like(state_locs)

        observation_reconstructions = self.bottler(self.decoder,(state, ))

        return observation_reconstructions, [state, state_locs, state_scales]

class VAE_1D_AuxReg(ModelInterface, jit.ScriptModule): 
    model_interface = M.vae_aux
    data_interface = D.dynamic
    def __init__(self, state_size: int, observation_dimensionality, action_dimensionality, action_dimensionality_conditional, encoder_args: dict, decoder_args: dict, aux_reg_args: dict, min_std_dev: float = 0.01,*args, **kwargs) -> None:
        super(VAE_1D_AuxReg, self).__init__()
        self.min_std_dev = min_std_dev
        self.bottler = model_utils.bottle
        state_size_c = action_dimensionality_conditional
        state_size_not_c = state_size - state_size_c
        
        state_size_c, state_size_not_c = model_utils.parse_cond_prior_args_for_latent_dims(self.model_interface, aux_regs=[aux_reg_args], original_latent_spaces_sizes=[state_size_c, state_size_not_c], action_dimensionalities=[action_dimensionality_conditional], cond_priors=[None])
        
        self.state_size_c = state_size_c
        self.state_size_not_c = state_size_not_c
        self.encoder = networks.Encoder_1D(state_size, input_dimensionality=observation_dimensionality, **encoder_args, )
        self.decoder = networks.Decoder_1D(state_size, input_dimensionality=observation_dimensionality, **decoder_args, )
        self.aux_reg = getattr(networks, aux_reg_args.pop("object"))(in_dim=state_size_c, out_dim=action_dimensionality_conditional, **aux_reg_args)


    def decode_state(self, states: torch.Tensor): 
        pass 

    def forward(self, observations: torch.Tensor, actions: torch.Tensor, *args, **kwargs): 
        state_locs, state_scales = self.bottler(self.encoder, (observations, ))
        state_scales = F.softplus(state_scales) + self.min_std_dev
        state = state_locs + state_scales*torch.rand_like(state_locs)

        state_c, state_not_c = state.split([self.state_size_c, self.state_size_not_c], -1)
        action_reconstructions = self.aux_reg(state_c)
        observation_reconstructions = self.bottler(self.decoder,( state, ))

        return observation_reconstructions, [state, state_locs, state_scales], action_reconstructions
