from pathlib import Path 
import sys 
import os 
PATH = Path.cwd() 
sys.path.insert(0, str(Path(PATH).parent))

from utils import load_model, shot_prediction, uncertainty_quantification_1

run_name, model_name = 'vaunted-pig-275', 'model'
# run_name, model_name = 'honorable-gull-235', 'model'
# run_name, model_name = 'adaptable-chimp-787', 204
model_interface, data_interface = load_model(run_name, model_name)

from common.interfaces import D, M
m_interface: M = model_interface.observational.model_interface
d_interface: D = model_interface.data_interface

train_indicies, valid_indicies, test_indicies = data_interface.train_dataset.indices, data_interface.valid_dataset.indices, data_interface.test_dataset.indices
actions_name_list = data_interface.dataset.filter_mps_names
shot_numbers = data_interface.dataset.shot_numbers
train_shots, valid_shots, test_shots = [sorted([shot_numbers[t] for t in idxs]) for idxs in [train_indicies, valid_indicies, test_indicies]]


"""
shot_num = train_shots[100]
observation_info, action_info, time, radius, batch_terminal, latent_space_information = shot_prediction(shot_num, model_interface, data_interface)

_observations, _obs_encoder, _obs_forward = observation_info
[obs_ls_sample, obs_ls_mu, obs_ls_var],  [trans_ls_sample, trans_ls_mu, trans_ls_var] = latent_space_information

# [obs_ls_sample, obs_ls_mu, obs_ls_var], [trans_ls_sample. trans_ls_mu, trans_ls_var] = latent_space_information
print(_observations.shape, trans_ls_mu.shape, radius.shape, obs_ls_mu.shape)
"""

import gradio as gr 
import torch
import numpy as np 
import matplotlib.pyplot as plt 

""" 
Latent space plotting, N sliders for N latent dims -> profile output
"""

def plot_profile_from_ls_vector(*zs_points): 
	plt.close('all')
	print(zs_points)
	zs_points = torch.tensor(zs_points)
	sample = zs_points.unsqueeze(0).unsqueeze(1).float()
	sample = sample.repeat((100, 1, 1))
	with torch.no_grad(): 
		sample_deco = model_interface.infer_from_state(sample)
	sample_deco = data_interface.dataset.denorm_profs(sample_deco).squeeze(1)
	fig, axs = plt.subplots()
	t_axs = axs.twinx()
	axs.plot(sample_deco[0, 0], color='black')
	t_axs.plot(sample_deco[0, 1], color='orange')
	return fig

def plot_profile_ls_dim(zs_dim): 
	if isinstance(zs_dim, str): 
		zs_dim = int(zs_dim)
	N_samples = 1000
	zs_points = torch.zeros((N_samples, 1, model_interface.observational.state_size))
	interped = torch.linspace(-5, 5, N_samples)
	zs_points[:, 0, zs_dim] = interped

	with torch.no_grad(): 
		sample_deco = model_interface.infer_from_state(zs_points)
	sample_deco = data_interface.dataset.denorm_profs(sample_deco).squeeze(1)
	radii = torch.arange(0, 200).unsqueeze(0)

	# colors = [(0, 0, )]
	
	fig, axs = plt.subplots(2, 1, sharex=True)
	axs[0].plot(radii.repeat((N_samples, 1)), sample_deco[:, 0])
	axs[1].plot(radii.repeat((N_samples, 1)), sample_deco[:, 1])
	return fig 

def plot_unc_rollout(shot_num): 
	pred_samp_mean, pred_samp_samp, obs, obs_recons, data_x, data_time = uncertainty_quantification_1(shot_num, model_interface, data_interface)
	rhos = [0.0, 0.4, 0.5, 0.9]
	fig, axs = plt.subplots(2, len(rhos), sharex=True, dpi=200, sharey='row')
	N_SAMPLES = pred_samp_samp.shape[1]
	for col_idx, rho in enumerate(rhos): 
		rho_idx = np.argmin(abs(data_x - rho))
		for i in [0, 1]: 
			axs[i, col_idx].plot(data_time.repeat(N_SAMPLES, 1).T, pred_samp_mean[:, :, i, rho_idx], color='black', zorder=20)
			axs[i, col_idx].plot(data_time, obs[:, 0, i, rho_idx], color='red', zorder=25)
			axs[i, col_idx].plot(data_time, obs_recons[:, 0, i, rho_idx], color='orange', zorder=23)
			axs[i, col_idx].plot(data_time.repeat(N_SAMPLES, 1).T, pred_samp_samp[:, :, i, rho_idx], color='dodgerblue', zorder=5, alpha=0.1)
			axs[i, col_idx].grid()
		axs[0, col_idx].set_title(r'$\rho$' + f'= {rho}')
	fig.subplots_adjust(hspace=0.01)
	fig.suptitle(shot_num)
	return fig

def plot_latent_rollout(shot_num): 
	observations, actions, time, data_x, batch_terminus, latent_space_information = shot_prediction(shot_num, model_interface, data_interface)
	[_, post_loc, post_scale], [_, prior_loc, prior_scale] = latent_space_information

	fig, axs = plt.subplots(model_interface.observational.state_size, 1, figsize=(5, 10), sharex=True, sharey=True)
	
	for ls_idx in range(model_interface.observational.state_size): 
		axs[ls_idx].fill_between(time, post_loc[:, ls_idx] + post_scale[:, ls_idx], post_loc[:, ls_idx] - post_scale[:, ls_idx], color='grey', alpha=0.5)
		axs[ls_idx].plot(time, post_loc[:, ls_idx], label='Observational', zorder=20)

		axs[ls_idx].fill_between(time, prior_loc[:, ls_idx] + post_scale[:, ls_idx], prior_loc[:, ls_idx] - prior_scale[:, ls_idx], color='orange', alpha=0.5)
		axs[ls_idx].plot(time, prior_loc[:, ls_idx], label='Forward', zorder=20)
		axs[ls_idx].set_ylabel(f'z-{ls_idx}')
		axs[ls_idx].grid()
	axs[0].legend()
	fig.subplots_adjust(hspace=0.0)
	fig.suptitle(shot_num)

	return fig 


title=f'{run_name} - {model_name}'
demo = gr.Blocks() 
with demo: 
	gr.Markdown(f"#{title}")
	with gr.Tab('Latent-Single'):
		with gr.Row(): 
			with gr.Column(): 
				inputs = [gr.Slider(value = 0, minimum=-10, maximum=10, label=f'z-{i}') for i in range(model_interface.observational.state_size)] 
			plot = gr.Plot()
			for inp in inputs: 
				inp.change(plot_profile_from_ls_vector, inputs, outputs=[plot])
		with gr.Row(): 
			with gr.Column(): 
				inputs_2 = gr.Radio(choices=[f'{i}' for i in range(model_interface.observational.state_size)]) 
			plot_2 = gr.Plot()
			inputs_2.change(plot_profile_ls_dim, inputs_2, outputs=[plot_2])
	with gr.Tab('Latent-Time'): 
		with gr.Row(): 
			with gr.Column(scale=1): 
				shot_dropdowns_ls = [gr.Dropdown(shot_list, value=shot_list[0], label=name) for name, shot_list in zip(['Train', 'Valid', 'Test'], [train_shots, valid_shots, test_shots])]
			with gr.Column(scale=5): 
				plot_ls_time = gr.Plot()
			for drop in shot_dropdowns_ls: 
				drop.change(plot_latent_rollout, drop, outputs=[plot_ls_time])
			

	with gr.Row(): 
		with gr.Column(scale=1): 
			unc_dropdowns = [gr.Dropdown(shot_list, value=shot_list[0], label=name) for name, shot_list in zip(['Train', 'Valid', 'Test'], [train_shots, valid_shots, test_shots])]
		with gr.Column(scale=5): 
			plot_unc_roll = gr.Plot()
		for unc_dropdown in unc_dropdowns: 
			unc_dropdown.change(plot_unc_rollout, unc_dropdown, outputs=[plot_unc_roll])

demo.launch()