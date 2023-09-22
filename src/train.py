import argparse 
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('-cf', '--config_file', type=str, default='base', help='name of configuration file!')
args = parser.parse_args() 

import os 
from configs import parsing

config_filename = os.path.join('./configs', f'{args.config_file}.json')
config_dict = parsing.get_config_file(config_filename)
print(config_dict)
d = {**vars(args), **config_dict}
args = argparse.Namespace(**d)


import data 
import models 
import trainers 
import torch 

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.trainer['device'] = torch.device('cuda')
    args.trainer['pin_memory'] = True
    args.trainer['nw'] = int(os.getenv('SLURM_CPUS_PER_TASK', 0))
    args.model['device'] = torch.device('cuda')
    args.data['device'] = torch.device('cuda')
else:
    torch.set_default_device('cpu')
    torch.manual_seed(args.seed)
    args.trainer['device'] = torch.device('cpu')
    args.trainer['pin_memory'] = False
    args.model['device'] = torch.device('cpu')
    args.data['device'] = torch.device('cpu')

torch.set_default_dtype(torch.float32)

if args.use_mlflow: 
    import mlflow 

data_interface = data.DatasetInterface(args.data.pop('return_type'), **args.data)# getattr(data, args.data.pop('object'))(data_path=args.data_path, **args.data)

args = parsing.set_model_config_based_on_dataset(args, data_interface)
model_interface = getattr(models, args.model.pop('object'))(**args.model)

param_list = model_interface.parameters()
optimizer=torch.optim.Adam(param_list, lr=args.trainer['learning_rate'], eps=1e-4)

print(f"Training on {args.trainer['device']}")
trainer = getattr(trainers, args.trainer.pop('object'))(**args.trainer, model=model_interface, data=data_interface, criterion=None, config={**args.trainer, **args.model, **args.data}, optimizer=optimizer, use_mlflow=args.use_mlflow, config_file=args.config_file, experiment_name=args.experiment_name)

trainer.train()
