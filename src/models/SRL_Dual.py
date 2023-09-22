from models.base import DualModelBaseInterface, ModelInterface
from common.interfaces import M, D 
from . import observational as obs_models
from . import forward as forw_models

import torch 
from torch import jit 
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal

class DualModelInterface(DualModelBaseInterface, jit.ScriptModule): 
    observational_model_interface = []
    transitional_model_interface = []
    data_interface = D.dynamic

    def __init__(self, observational_model: dict, transitional_model: dict, **kwargs):
        super().__init__()
        self.observational: ModelInterface = getattr(obs_models, observational_model.pop('object'))(**observational_model, **kwargs)
        self.transitional: ModelInterface = getattr(forw_models, transitional_model.pop('object'))(**transitional_model, **kwargs)

    def forward_observational(self, observations: torch.Tensor, actions: torch.Tensor):
        return self.observational(observations, actions) 

    def infer_from_state(self, state: torch.Tensor): 
        return self.observational.decode_state(state)

    def pulse_inference(self, states_all: torch.Tensor, actions: torch.Tensor, starting_T: int = 0): 
        predicted_states, predicted_states_mu, predicted_states_std_devs = self.transitional.pulse_inference(states_all, actions, starting_T)
        predicted_recons = self.infer_from_state(predicted_states)
        return predicted_recons, predicted_states, [predicted_states_mu, predicted_states_std_devs]

    def forward_transitional(self, state: torch.Tensor, state_mu: torch.Tensor,state_std_dev: torch.Tensor, actions: torch.Tensor, padding_bool: torch.Tensor, starting_T: int = 1, num_rollout: int=0, max_samples: int = 5): 
        return self.transitional.forward(state, state_mu, state_std_dev, actions, padding_bool, starting_T=starting_T, num_rollout=num_rollout, max_samples=max_samples)