from .base import ModelInterface
from common.interfaces import M, D
from torch import nn, jit 
import torch 
import torch.nn.functional as F
from typing import List, Union, Tuple

class LinearForwardModel(ModelInterface, jit.ScriptModule): 
    model_interface = M.linear
    data_interface = D.dynamic
    __constants__ = ['state_size', 'action_size', 'min_std_dev']
    def __init__(self, state_size: int, action_dimensionality: int, hidden_layers: Union[int, List[int]], pushforward_trick: str, min_std_dev: float = 0.01, *args, **kwargs): 
        super(LinearForwardModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_dimensionality
        self.min_std_dev = min_std_dev
        self.pushforward = pushforward_trick
        if isinstance(hidden_layers, int):
            self.forward_block = nn.Linear(self.state_size+self.action_size, hidden_layers)
            out_dim = hidden_layers
        elif isinstance(hidden_layers, list): 
            forward_block = torch.nn.ModuleList()
            in_dim = self.action_size+self.state_size
            for out_dim in hidden_layers: 
                forward_block.append(nn.Linear(in_dim, out_dim))
                forward_block.append(nn.Tanh())
                in_dim = out_dim 
        self.fc_state_transition = nn.Linear(out_dim, 2*self.state_size)
    
    def forward_single(self, state_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        hidden = self.forward_block(state_action)
        loc, scale = torch.chunk(self.fc_state_transition(hidden), 2, dim=-1)
        scale = F.softplus(scale) + self.min_std_dev
        state = loc + scale * torch.rand_like(loc)
        return state, loc, scale
    
    def forward(self, states: torch.Tensor, state_locs: torch.Tensor, state_scales: torch.Tensor, actions: torch.Tensor, padding_bool: torch.Tensor, num_rollout: int=0, **kwargs): 
        T : int = actions.size(0) - num_rollout # total time to loop through 
        posterior_states, posterior_means, posterior_std_devs, prior_states, prior_means, prior_std_devs, padding_states = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T

        for t in range(T): 
            _t_roll: int = 0
            _state, _action = states[t], actions[t]
            _state_action = torch.cat([_state, _action], dim=-1)
            if num_rollout > 1 and self.pushforward == 'last': 
                with torch.no_grad(): 
                    _state, _locs, _scales = self.forward_single(_state_action)
            else: 
                _state, _locs, _scales = self.forward_single(_state_action)
            for _t_roll in range(1, num_rollout): # This will not run for num_rollout = 1         
                if self.pushforward == 'last' and _t_roll != num_rollout -1 : 
                    with torch.no_grad(): 
                        _state, _action = _state, actions[t +_t_roll]
                        _state_action = torch.cat([_state, _action], dim=-1)
                        _state, _locs, _scales = self.forward_single(_state_action)
                else: 
                    _state, _action = _state, actions[t +_t_roll]
                    _state_action = torch.cat([_state, _action], dim=-1)
                    _state, _locs, _scales = self.forward_single(_state_action)                        
                    
            prior_states[t] = _state 
            prior_means[t], prior_std_devs[t] = _locs, _scales

            padding_states[t] = padding_bool[t + _t_roll + 1]
            posterior_states[t] = states[t + _t_roll + 1] # state where transition model should be compared to 
            posterior_means[t], posterior_std_devs[t] = state_locs[t + _t_roll + 1], state_scales[t + _t_roll + 1]

        return [torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0), torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0), torch.stack(posterior_std_devs[1:], dim=0), torch.stack(padding_states[1:], dim=0)]
    
    def pulse_inference(self, states_all: List[torch.Tensor], actions, starting_T: int = 0): 
        T = actions.size(0)
        states, states_mu, states_std_devs = states_all
        predicted_states, predicted_states_mu, predicted_states_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
        _state = states[0]
        predicted_states[0], predicted_states_mu[0], predicted_states_std_devs[0] = _state, states_mu[0], states_std_devs[0]
        if starting_T > 0: 
            print(f'Delay by {starting_T}')


        for n in range(starting_T): 
            _state_action = torch.cat([states[n], actions[n]], dim=-1)
            _state, _locs, _scales = self.forward_single(_state_action)
            predicted_states[n+1] = _state
            predicted_states_mu[n+1], predicted_states_std_devs[n+1] = _locs, _scales

        for t in range(starting_T, T-1): 
            _state_action = torch.cat([_state, actions[t]], dim=-1)
            _state, _locs, _scales = self.forward_single(_state_action)

            predicted_states[t+1] = _state
            predicted_states_mu[t+1], predicted_states_std_devs[t+1] = _locs, _scales
        
        predicted_states, predicted_states_mu, predicted_states_std_devs = torch.stack(predicted_states, dim=0), torch.stack(predicted_states_mu, dim=0), torch.stack(predicted_states_std_devs, dim=0)
        return predicted_states, predicted_states_mu, predicted_states_std_devs
