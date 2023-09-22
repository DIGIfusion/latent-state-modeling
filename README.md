# latent-state-modeling

This repo may be updated in the future.

Should start by cloning this repo, making a fresh virtual environment, and installing libraries in `requirements.txt`.

### Raw Data Access
Raw data will be provided upon reasonable request, **iff** you i) are a EUROfusion member, and ii) have access to JET and ASDEX-U machines. 

If both are fufilled, then the location of the data on both JET/AUG servers can be supplied, from which you can store and run the below scripts. 

## Running experiments

### Size Scaling 

To reproduce the size scaling experiments found in [A.E. Järvinen, citation TBD], you must build the data based on the transformations provided in the paper. This can be done by running the bash script `./build_data_multi_machine_multi_machine.local`, and changing the raw data directory names in the `.local` file to match those that you downloaded, and the desired array output directory (`$ARRAY_FOLDER`) to where you would like. 


The configuration files for the experiments in [A.E. Järvinen, citation TBD] are found in the `/configs/` directory: 
1. `BASE_VAE`, replace with hyperparameters from [Table #X from Paper ]
2. `DIVA_MultiMachine`, replace with hyperparameters from [Table #X from Paper ]
3. `SSVAE_Split_Variable`, replace with hyperparameters from [Table #X from Paper ]

After building the data, change the `data/data_path` variable in the config file to the location of the stored arrays (`$ARRAY_FOLDER`). 


To train the models, run `python3 train.py -cf=name_of_config_file`, where name of config file can be `BASE_VAE_SIZE_SCALING_model_1` for example. 

The models will be saved under the directory `saved_models` under a directory specified by the `experiment-name` variable in the config file.


### Dual Model 

To reproduce the Dual model experiments [A. Kit citation TBD], follow similar steps, but with: 

- `build_data.local` for the data building of just AUG data. 
- `SRL_1:` config for the base SRL modeling, 
- `SRL_w_aux_reg`: config for SRL model with linear subspace w.r.t to chosen parameters.

## Plotting

To visualize the results, two example notebooks are given in the `notebooks` dirs. 
## License
TBD 