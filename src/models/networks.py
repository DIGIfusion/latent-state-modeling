import torch 
from torch import nn, jit 
import torch.nn.functional as F 
from typing import List, Union, Tuple, Optional
from .model_utils import parse_convolution_argument 

class Encoder_1D(nn.Module): 
    __constants__ = ['state_size']
    def __init__(self, state_size: int, input_dimensionality: Tuple[int, int], activation_function: str='ReLU', channel_dims: List[int] =[4, 8, 16, 32], kernel_sizes: Union[List[int], int]= 4, stride_sizes: Union[List[int], int]=2, *args, **kwargs) -> None:
        super(Encoder_1D, self).__init__()
        self.state_size = state_size 
        self.act_fn = getattr(nn, activation_function)

        in_ch_dim, in_spatial_dim = input_dimensionality
        sample = torch.ones((1, in_ch_dim, in_spatial_dim))

        # Check if arguments correct
        num_layers = len(channel_dims)
        kernel_sizes = parse_convolution_argument(kernel_sizes, num_layers, arg_name='Kernel Sizes')
        stride_sizes = parse_convolution_argument(stride_sizes, num_layers, arg_name='Stride Sizes')
        
        self.conv_block = nn.ModuleList()
        for n, (out_ch_dim, kernel_size, stride_size) in enumerate(zip(channel_dims, kernel_sizes, stride_sizes)): 
            self.conv_block.append(nn.Conv1d(in_ch_dim, out_ch_dim, kernel_size=kernel_size, stride=stride_size))
            if n != num_layers - 1: 
                self.conv_block.append(self.act_fn())
            in_ch_dim = out_ch_dim
        # get the output size :D
        for lay in self.conv_block:
            sample = lay(sample)
        self.hidden_output_size = sample.shape.numel()
        self.locs = nn.Linear(self.hidden_output_size, self.state_size)
        self.scales = nn.Linear(self.hidden_output_size, self.state_size)

    def forward(self, observation: torch.Tensor): 
        for lay in self.conv_block: 
            observation = lay(observation)
        observation = observation.view(-1, self.hidden_output_size)
        return self.locs(observation), self.scales(observation)
    

class Decoder_1D(nn.Module): 
    def __init__(self, state_size: int, embedding_size: int, input_dimensionality: Tuple[int, int], activation_function: str='ReLU', channel_dims: List[int] =[16,8,4,2], kernel_sizes: Union[List[int], int]= [5,5,6,6], stride_sizes: Union[List[int], int]=3, output_clamping: Optional[torch.Tensor] = None, *args, **kwargs) -> None:
        # the channel dims must include the final output channel dimensionality
        super(Decoder_1D, self).__init__()
        self.act_fn = getattr(nn, activation_function)
        self.state_size = state_size 
        self.embedding_size = embedding_size
        self.out_ch_dim, self.out_spatial_dim = input_dimensionality

        num_layers = len(channel_dims)
        kernel_sizes = parse_convolution_argument(kernel_sizes, num_layers, arg_name='Kernel Sizes')
        stride_sizes = parse_convolution_argument(stride_sizes, num_layers, arg_name='Stride Sizes')

        self.fc_embedding = nn.Linear(self.state_size, self.embedding_size)
        self.trans_conv_block = nn.ModuleList()
        

        in_ch_dim = self.embedding_size
        for n, (out_ch_dim, kernel_size, stride_size) in enumerate(zip(channel_dims, kernel_sizes, stride_sizes)): 
            self.trans_conv_block.append(nn.ConvTranspose1d(in_ch_dim, out_ch_dim, kernel_size=kernel_size, stride=stride_size))
            self.trans_conv_block.append(self.act_fn())
            in_ch_dim = out_ch_dim

        sample = self.fc_embedding(torch.ones((1, state_size)))
        sample = sample.view(-1, self.embedding_size, 1)
        for lay in self.trans_conv_block: 
            sample = lay(sample)
        hidden_output_size = sample.shape[-1]
        
        self.fc_out = nn.Linear(hidden_output_size, self.out_spatial_dim)
        self.output_clamping = output_clamping

    def forward(self, state: torch.Tensor)-> torch.Tensor: 
        state = self.fc_embedding(state)
        state = state.view(-1, self.embedding_size, 1)
        for lay in self.trans_conv_block: 
            state = lay(state)
        observation = self.fc_out(state)
        if self.output_clamping is not None: 
            observation = torch.clamp(observation, min=self.output_clamping, max=None)
        return observation

class Prior(nn.Module): 
    def __init__(self, in_dim: int, out_dim: int, hidden_sizes: Union[List[int], int], activation_function: str='ReLU', **kwargs) -> None:
        super(Prior, self).__init__()
        self.act_fn = getattr(nn, activation_function)
        self.reg_block = nn.ModuleList()
        if isinstance(hidden_sizes, int): 
            hidden_sizes = [hidden_sizes]

        for h_dim in hidden_sizes: 
            self.reg_block.append(nn.Linear(in_dim, h_dim))
            self.reg_block.append(self.act_fn())
            in_dim = h_dim
        # self.reg_block.append(nn.Linear(in_dim, out_dim))
        self.locs = nn.Linear(in_dim, out_dim)
        self.scales = nn.Linear(in_dim, out_dim)
    def forward(self, state: torch.Tensor) -> torch.Tensor: 
        for lay in self.reg_block: 
            state = lay(state)
        
        return self.locs(state), self.scales(state)         
class Regressor(nn.Module): 
    def __init__(self, in_dim: int, out_dim: int, hidden_sizes: Union[List[int], int], activation_function: str='ReLU', **kwargs) -> None:
        super(Regressor, self).__init__()
        self.act_fn = getattr(nn, activation_function)
        self.reg_block = nn.ModuleList()
        if isinstance(hidden_sizes, int): 
            hidden_sizes = [hidden_sizes]

        for h_dim in hidden_sizes: 
            self.reg_block.append(nn.Linear(in_dim, h_dim))
            self.reg_block.append(self.act_fn())
            in_dim = h_dim
        self.reg_block.append(nn.Linear(in_dim, out_dim))

    def forward(self, state: torch.Tensor) -> torch.Tensor: 
        for lay in self.reg_block: 
            state = lay(state)
        return state         
class Diagonal(nn.Module):
    def __init__(self, dim: int):
        super(Diagonal, self).__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(self.dim))
        self.bias = nn.Parameter(torch.zeros(self.dim))
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        return x * self.weight + self.bias

class DiagonalRegressor(nn.Module): 
    def __init__(self, dim: int, *args, **kwargs) -> None:
        super(DiagonalRegressor, self).__init__()
        self.dim = dim 
        self.diag = Diagonal(self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.diag(x) 
        return x
    
class DiagonalPrior(nn.Module): 
    def __init__(self, dim,  *args, **kwargs) -> None:
        super(DiagonalPrior, self).__init__()
        self.diag_loc = Diagonal(dim)
        self.diag_scale = Diagonal(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.diag_loc(x), self.diag_scale(x)
