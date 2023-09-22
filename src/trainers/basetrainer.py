# from models.base import ModelInterface, DualModelBaseInterface
from data.interface import DatasetInterface
from typing import Callable, List, Tuple, Union, Dict
import argparse 
from common.interfaces import M, D
from abc import ABCMeta, abstractmethod
import torch 
import torch.nn as nn 
from torch import optim 
from torch.utils.data import DataLoader
import timeit
import os 
import pathlib
import shutil
try: 
    import mlflow 
    import matplotlib.pyplot as plt 
except ImportError as e: 
    use_mlflow = False

class TrainInterface(metaclass=ABCMeta):
    def __init__(
        self,
        model,
        data: DatasetInterface,
        criterion: Callable,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler,
        config: argparse.Namespace = None,
        experiment_name: str = "STANDALONE",
        use_mlflow=True,
        mlflow_kwargs=None,
        mlflow_config_dict=None,
        **kwargs,
    ):
        self.model = model
        self.data = data 
        self.use_mlflow = use_mlflow
        self.experiment_name = os.path.abspath(os.path.join('./saved_models', experiment_name))
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler

        self.max_train_batches = kwargs.get('max_train_batches', float("inf"))
        self.max_test_batches = kwargs.get('max_test_batches', float("inf"))
        self.print_interval = kwargs.get('print_interval', 1)
        self.device = kwargs.get('device')
        self.nw = kwargs.get('nw', 0)
        self.test_interval = kwargs.get('test_interval', 10)
        self.batch_size = kwargs.get('batch_size', 8)
        self.num_epochs = kwargs.get('num_epochs', 10)
        self.config_file = kwargs.get('config_file', '')
        self.create_figure_dir()
        self.config=config
        self.log_artifacts()
    def __repr__(self):
        return self.__class__.__name__

    def log_artifacts(self,): 
        if self.use_mlflow: 
            mlflow.set_experiment(self.experiment_name)
            with mlflow.start_run() as run: 
                mlflow.log_artifact(f'./configs/{self.config_file}.json', 'configs')
                mlflow.log_artifact(os.path.join(self.config["data_path"], self.config['transform_file']), 'configs')
                # TODO: dump as dict
                for key, value in self.config.items():
                    if key in ['scale_observation_decoder', 'scale_cond_reg', 'observational_model', 'transitional_model']:
                        pass 
                    else:  
                        print(key, value)
                        mlflow.log_param(key, value)

        else: 
            import shutil 
            savedir = os.path.join(self.experiment_name)
            if not os.path.exists(savedir): 
                os.mkdir(savedir)
            
            shutil.copyfile(f'./configs/{self.config_file}.json', os.path.join(savedir, 'config.json'))
            shutil.copyfile(os.path.join(self.config["data_path"], self.config['transform_file']), os.path.join(savedir, 'transformations.pickle'))
    @property
    @abstractmethod
    def data_interface(self) -> List[D]:
        """A list for storing what data interfaces the training method is compatible with"""
        raise NotImplementedError("data_interface not set!")
    @property
    @abstractmethod
    def model_interface(self) -> List[M]:
        """A list for storing what model interfaces the training method is compatible with"""
        raise NotImplementedError("model_interface not set!")
    def create_figure_dir(self): 
        
        if not os.path.exists(pathlib.Path(self.experiment_name).parent):
            os.mkdir(pathlib.Path(self.experiment_name).parent) 
        if not os.path.exists(self.experiment_name): 
            os.mkdir(self.experiment_name)
        figure_dir = os.path.join(self.experiment_name, 'figures')
        if not os.path.exists(figure_dir):
            os.mkdir(figure_dir)
        
        # os.path.join(self.experiment_name, save_name)

    def get_parameters(self):
        return self.model.parameters()

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_lr_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def save_model(self, save_name): 
        if self.use_mlflow == True: 
            mlflow.pytorch.log_model(self.model, save_name) 
            print(f"Saved model at mlflow at {save_name}")
        else: 
            filename, ext = os.path.splitext(save_name)
            if ext == '':  # default extension
                ext = '.pt'
            save_name = filename + ext
            if isinstance(self.model, torch.jit.ScriptModule):
                torch.jit.save(self.model, os.path.join(self.experiment_name, save_name))
            else: 
                torch.save(self.model, os.path.join(self.experiment_name, save_name))
            print(f'Not using mlflow, saved model at {self.experiment_name}/{save_name}')

    def train_step(self, batch: Tuple, epoch, batch_idx, loader,) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method that should be implemented for a TrainInterface class. This method is called by the train method of the
        TrainInterface class that has already been implemented. Only the forward pass's loss calculation
        should be implemented; the backward pass and learning step is taken care of outside this method.
        Args:
            batch: tuple with a batch of data -- you should implement here how to get model input and labels from the data.
                    This will depend on the type of training (e.g. temporal bundling, pushforward, ...) and model
                    (graph, grid, ...), and what kind of data is available. Typically this is the channels, spatial
                    coordinates and possibly PDE parameters. If the batch tuple contains tensors, these have been moved
                     to the specified device already for the forward pass.
            epoch: index of epoch
            batch_idx: index of batch
            loader: the DataLoader used for training
        Returns:
            Zero-dimensional tensor with loss value for this batch, a dictionary with various losses predictions for this batch
        """
        raise NotImplementedError("The method train_step should be implemented!")
        return loss_of_this_batch, loss_dict_on_this_batch, preds

    def test_step(self, batch: Tuple, batch_idx: int, use_train_loss_calc=True, include_data=False, **kwargs) -> Tuple[Union[torch.Tensor, float], dict]:
        """
        To be implemented for a specific testing strategy
        Args:
            batch: batch with data
            batch_idx: index of the batch
            use_train_loss_calc: boolean indicating if train_step is to be used - will be set to True if test_step
                raises a NotImplementedError.
            include_data: boolean indicating whether to include the ground truth + simulated data in the output dict
        """
        if False:
            raise ValueError("include_data is only supported when implemented in test_step")
        else:
            loss, loss_dict, preds = self.train_step(batch, epoch=0, batch_idx=batch_idx, **kwargs)

        return loss, loss_dict, preds

    def plot_results(self, batch_on_device, datatoplot, epoch, shot_nums, batch_idx):
        print('Plotting not implemented yet!')
        return None
        
    def __call__(self):
        self.train()

    def get_dataloaders(self):
        """
        Helper function to retrieve dataloaders based on the specified config.
        Returns:
            DataLoader train data
            DataLoader valid data
            DataLoader test data
        """
        persistent_workers = self.nw > 0
        pin_memory = True
    
        dataloader_kwargs = dict(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.nw,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
        )

        train_loader = DataLoader(self.data.train, **dataloader_kwargs)
        valid_loader = DataLoader(self.data.valid, **dataloader_kwargs)
        test_loader = DataLoader(self.data.test, **dataloader_kwargs)

        return train_loader, valid_loader, test_loader

    def train(self) -> Tuple[List[Union[float, torch.Tensor]], Dict[str, List[Union[float, torch.Tensor]]], Dict[str, List[Dict]]]:
        """
        Trains the model with which the class was initialized on the dataset with which it was initialized
        Args:
        Returns:
            list of training losses per epoch, list of val losses per epoch, list of (dict of val stats) per epoch
        """
        device = self.device
        self.model = self.model.to(device)
        train_loader, valid_loader, test_loader = self.get_dataloaders()
        print('*****Starting Training*****')

        train_losses = []
        time_start = timeit.default_timer()
        for epoch in range(self.num_epochs):
            train_loss = self.train_one_epoch(train_loader, epoch)
            train_losses.append(train_loss)
            if (epoch + 1) % self.test_interval == 0: 
                val_loss, val_stats = self.test(valid_loader, epoch=epoch)
        self.save_model('final')
        print("Total training time {:.4}s".format(timeit.default_timer() - time_start))
        return None 

    def test(self, loader: torch.utils.data.DataLoader, use_train_loss_calc=True, include_data=True, epoch=0) -> Union[Tuple[Union[float, torch.Tensor], dict], Tuple[Union[float, torch.Tensor], dict, Tuple[torch.Tensor, list]]]:
        print('Testing!')
        test_kwargs = {}
        device = self.device 
        self.model.eval()
        loss = 0
        # other_metrics = {'observation_reconstruction': 0.0, 'observation_from_mps': 0.0, 'kl': 0.0}
        n_total = 0
        
        with torch.no_grad(): 
            for batch_idx, batch in enumerate(loader): 
                *batch, shot_nums = batch
                batch_on_device = tuple(item.to(device).float() for item in batch)

                batch_size = batch_on_device[0][1].shape[1]
                test_step_out = self.test_step(batch_on_device, use_train_loss_calc=False, loader=loader, batch_idx=batch_idx, include_data=include_data, batch_shot_numbers=shot_nums)
                if include_data: 
                    batch_loss, batch_loss_dict, datatoplot = test_step_out 
                    self.plot_results(batch_on_device, datatoplot, epoch, shot_nums, batch_idx)
                else: 
                    batch_loss, batch_loss_dict = test_step_out
                
                loss += batch_loss*batch_size 
                n_total += batch_size
                for k, v in batch_loss_dict.items(): 
                    val = test_kwargs.get(k, 0.0)
                    test_kwargs[k] = val + v*batch_size
                    # if k in test_kwargs: 
                    #     test_kwargs[k] += v*batch_size 
                    # else: 
                    #     test_kwargs[k] = v*batch_size 
                if batch_idx >= self.max_test_batches - 1:
                    break
        loss = loss / n_total 
        other_metrics = {k: v / n_total for k, v in test_kwargs.items()}
        if self.use_mlflow: 
            mlflow.log_metric('val_loss', loss, step=epoch)
            for k, v in other_metrics.items(): 
                mlflow.log_metric(f'val_{k}_loss', v, step=epoch)
            self.save_model(f'{epoch}')
        else: 
            self.save_model(f'{epoch}')
        print(f'\tTest results {epoch}')
        for key, value in other_metrics.items(): 
            print(f'\t\t{key}:{value:.4}')
        return loss, other_metrics

    def train_one_epoch(self, loader: torch.utils.data.DataLoader, epoch) -> float:
        """
        method that loops over all batches for training in an epoch.
        Args:
            loader: Pytorch Dataloader that returns x, labels tensors for train_batch method
            epoch: index of epoch
        Returns:
            float of loss value for this epoch
        """
        self.model.train()
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler
        device = self.device
        total_loss = 0
        loss_dict_epoch = {}
        for batch_idx, batch in enumerate(loader):
            *batch, shot_nums = batch
            batch_on_device = tuple(item.to(device).float() for item in batch)
            optimizer.zero_grad()
            loss, loss_dict, preds = self.train_step(batch_on_device, epoch, batch_idx, loader=loader)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 10, norm_type=2)
            optimizer.step()
            batch_size = batch_on_device[0].shape[1]
            total_loss += loss.detach() / batch_size
            for key, value in loss_dict.items(): 
                prev_total_loss = loss_dict_epoch.get(key, 0.0)
                prev_total_loss += value.detach() 
                loss_dict_epoch[key] = prev_total_loss
            if batch_idx >= self.max_train_batches:
                break
        total_loss = total_loss / len(loader)
        if self.use_mlflow: 
            mlflow.log_metric('train_loss', total_loss, step=epoch)
            for key, value in loss_dict_epoch.items(): 
                value /= len(loader)
                mlflow.log_metric(f'train_{key}_loss', value, step=epoch)
        if (epoch + 1) % self.print_interval == 0: 
            print(f'Train Results Epoch {epoch}')
            for key, value in loss_dict_epoch.items(): 
                print(f'\t {key}:{value:.4} {((value / batch_size) / len(loader)) / total_loss.detach().item():.2}')
            pass 

        if lr_scheduler is not None:
            if (epoch + 1) % self.lr_step_interval == 0:
                lr_scheduler.step()
        return total_loss