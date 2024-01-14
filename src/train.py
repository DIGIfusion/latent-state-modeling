import argparse 
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('-cf', '--config_file', type=str, default='base', help='name of configuration file!')
args = parser.parse_args() 

import os 
from configs import parsing

config_filename = os.path.join('./configs', f'{args.config_file}.json')
config_dict = parsing.get_config_file(config_filename)
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

if args.use_mlflow: 
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run() as run: 
        mlflow.log_artifact(f'./configs/{args.config_file}.json', 'configs')
        mlflow.log_artifact(os.path.join(args.data_path, args.data['transform_file']), 'configs')
        mlflow.log_artifact(os.path.join(args.data_path, 'DATA_CONFIG.yaml'), 'configs')
        for key, value in {**args.trainer, **args.model, **args.data, **{'data_path': args.data_path}}.items():
            if key in ['scale_observation_decoder', 'scale_cond_reg', 'observational_model', 'transitional_model']:
                pass
            else:
                print(key, value)
                mlflow.log_param(key, value)
        trainer.train()
else: 
    import shutil 
    savedir = os.path.join('./saved_models', args.experiment_name)
    if not os.path.exists(savedir): 
        os.mkdir(savedir)
    shutil.copyfile(f'./configs/{args.config_file}.json', os.path.join(savedir, 'config.json'))
    shutil.copyfile(os.path.join(args.data['data_path'], args.data['transform_file']), os.path.join(savedir, 'transformations.pickle'))
    shutil.copyfile(os.path.join(args.data['data_path'], args.data['transform_file']), os.path.join(savedir, 'transformations.pickle'))
    shutil.copyfile(os.path.join(args.data['data_path'], 'DATA_CONFIG.yaml'), os.path.join(savedir, 'DATA_CONFIG.yaml'))
    trainer.train()
