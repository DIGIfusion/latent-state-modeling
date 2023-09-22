import torch 
from abc import abstractmethod, ABCMeta
from common.interfaces import D, M
from typing import List

class ModelInterface(torch.nn.Module): 
    @property 
    @abstractmethod
    def model_interface(self) -> M: 
        raise NotImplementedError('model_interface is not set')

    @property 
    @abstractmethod
    def data_interface(self) -> List[D]: 
        return []
    

class DualModelBaseInterface(torch.nn.Module): 
    @property 
    @abstractmethod
    def observational_model_interface(self) -> List[M]: 
        raise NotImplementedError('model_interface is not set')
    
    @property 
    @abstractmethod
    def transitional_model_interface(self) -> List[M]: 
        raise NotImplementedError('model_interface is not set')
    
    @property 
    @abstractmethod
    def data_interface(self) -> List[D]: 
        return []
    