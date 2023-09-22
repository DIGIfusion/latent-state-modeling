import aug_sfutils
import pickle 
import os 
from typing import List, Union 
import argparse 

from scipy.interpolate import interp1d
import numpy as np 
import torch 
import psutil 
from multiprocessing import Pool
import pandas as pd

from scipy.constants import mu_0, m_e, m_p, e

def clamp_power_vars(profiles:torch.Tensor, radii: torch.Tensor, mps, time) -> List[Union[torch.Tensor, np.ndarray]]:
    """ Clamp negative values of power to 0"""
    pow_idxs = [relevant_mp_columns.index(name) for name in [ 'PNBI_TOT', 'PICR_TOT','PECR_TOT', 'P_OH',] if name in relevant_mp_columns]
    for idx in pow_idxs: 
        mps[:, idx] = torch.clamp(mps[:, idx], min=0.0, max=None)
    return profiles, radii, mps, time

def convert_raw_file_to_tensor(filename: str, relevant_mp_columns: List[str],
                               additional_feature_engineering_columns: List[str],
                               mapping_functions: List[str],
                               limitcoretemp = True,
                               limitheatpower = True,
                               limitmaxslices = True,
                               max_pressure = True,
                               limitmprange = True,
                               limit_q95_and_correct_for_kappa = True) -> List[torch.Tensor]:
    shot_num: str = filename.split('/')[-1]
    device_name = 'AUG' if int(shot_num) < 70000 else 'JET'
    with open(filename, 'rb') as file: 
        pulse_dict = pickle.load(file)
    # TODO: IF JET, ADD ECRH 
    profile_data, mp_data, journal_data = \
        pulse_dict['profiles'], pulse_dict['machine_parameters'], pulse_dict['journal']
    negative_byte = b''
    if device_name == 'AUG':
        if pulse_dict['journal']['b_hmod'] == negative_byte: 
            return [None]*4
    if 'JET' in folder_name: 
        mp_data['PECR_TOT'] = {}
        mp_data['PECR_TOT']['time'], mp_data['PECR_TOT']['data'] = 'NO ECRH USED', None
        shot_loc_pdb = jet_pdb[jet_pdb['shot'] == int(shot_num)].iloc[0]
    if device_name == 'AUG': 
        journal_val: str = pulse_dict['journal']['impspez']
        if journal_val != negative_byte: 
            return [None]*4
        if pulse_dict['journal']['gas_h'] != negative_byte or pulse_dict['journal']['gas_d'] == negative_byte or pulse_dict['journal']['gas_he'] != negative_byte: 
            return [None]*4
        if  b'RMP' in pulse_dict['journal']['remarks'] or b'RMP' in pulse_dict['journal']['program']:
            return [None]*4
        if  b'pellet' in pulse_dict['journal']['remarks'] or b'pellet' in pulse_dict['journal']['program']:
            return [None]*4
        if  b'Pellet' in pulse_dict['journal']['remarks'] or b'Pellet' in pulse_dict['journal']['program']:
            return [None]*4
    elif device_name == 'JET':
        if int(shot_num) < 80000:
            return [None]*4 
        if shot_loc_pdb['flowrateofseededimpurity10^22(e/s)'] > 0.05e22: 
            return [None]*4
        if shot_loc_pdb['FLAG:HYDROGEN'] > 0.0 or shot_loc_pdb['FLAG:HeJET-C'] > 0.0: 
            return [None]*4 
        if shot_loc_pdb['FLAG:Kicks'] > 0.0 or shot_loc_pdb['FLAG:RMP'] > 0.0 or shot_loc_pdb['FLAG:pellets'] > 0.0:
            return [None]*4    
    if profile_data['ne'] is None or profile_data['Te'] is None or profile_data['radius'] is None: 
        raise ValueError('Some profiles dont exists here', shot_num)
    if 'AUG' in folder_name: 
        ida_times, ne, te, radius = \
            torch.from_numpy(profile_data['time']), torch.from_numpy(profile_data['ne'].T), \
            torch.from_numpy(profile_data['Te'].T), torch.from_numpy(profile_data['radius'].T)
    else:
        ida_times, ne, te, radius = \
            torch.from_numpy(profile_data['time']), torch.from_numpy(profile_data['ne']), \
            torch.from_numpy(profile_data['Te']), torch.from_numpy(profile_data['radius'])
    profiles = torch.stack((ne, te), 1)
    if not mp_data.get('PNBI_TOT'):
                raise ValueError
    MP_TIME_LIST = \
        torch.Tensor([(min(mp_data[key]['time']),
                       max(mp_data[key]['time'])) for key in relevant_mp_columns \
                      if isinstance(mp_data[key], dict) and not isinstance(mp_data[key]['time'], str)])
    
    MP_OBSERVATIONAL_END_TIME, MP_OBSERVATIONAL_START_TIME = \
        MP_TIME_LIST.min(0)[0][1], torch.max(MP_TIME_LIST, 0)[0][0]
    IDA_OBSERVATIONAL_END_TIME, IDA_OBSERVATIONAL_START_TIME = ida_times[-1], ida_times[0]
    
    t1, t2 = max(MP_OBSERVATIONAL_START_TIME, IDA_OBSERVATIONAL_START_TIME), \
        min(MP_OBSERVATIONAL_END_TIME, IDA_OBSERVATIONAL_END_TIME)
    
    relevant_time_windows_bool: torch.Tensor = torch.logical_and(ida_times > t1, ida_times < t2)
    #print('Limiting core temp')
    #if limitcoretemp:
        #relevant_time_windows_bool: torch.Tensor = torch.logical_and(te[:,0] > 2000, te[:,0] > 2000)
    #    relevant_time_windows_bool: torch.Tensor = torch.logical_and(te[:,0] > 1000, te[:,0] > 1000)
    #print('Applying limits')
    #print('Limit based on heating power')
    if limitheatpower:
        relevant_profiles = profiles[relevant_time_windows_bool]
        relevant_time_windows: torch.Tensor = ida_times[relevant_time_windows_bool]
        relevant_radii = radius[relevant_time_windows_bool]
        heatingparams = torch.empty((len(relevant_profiles), 4))
        counter = 0
        for mp_idx, key in enumerate(relevant_mp_columns):
            if key in ['PNBI_TOT', 'PICR_TOT', 'PECR_TOT', 'P_OH']:
                relevant_mp_vals = torch.zeros(len(relevant_profiles))
                mp_raw_data, mp_raw_time = mp_data[key]['data'], mp_data[key]['time']
                if mp_raw_time is None or isinstance(mp_raw_time, str):
                    pass
                else:
                    f = interp1d(mp_raw_time, mp_raw_data)
                    relevant_mp_vals = torch.from_numpy(f(relevant_time_windows))
                heatingparams[:,counter] = relevant_mp_vals
                counter += 1
                                                                              
        power = torch.sum(heatingparams, dim=1)
        if 'JET' in folder_name:
            relevant_time_windows_bool: torch.Tensor = torch.logical_and(power > 3e6, power > 3e6)
        else:
            relevant_time_windows_bool: torch.Tensor = torch.logical_and(power > 2e6, power > 2e6)
        relevant_profiles = relevant_profiles[relevant_time_windows_bool]
        relevant_time_windows: torch.Tensor = relevant_time_windows[relevant_time_windows_bool]
        relevant_radii = relevant_radii[relevant_time_windows_bool]
    else:
        relevant_time_windows: torch.Tensor = ida_times[relevant_time_windows_bool]
        relevant_profiles = profiles[relevant_time_windows_bool]
        relevant_radii = radius[relevant_time_windows_bool]
    if limit_q95_and_correct_for_kappa:
        checkparams = torch.empty((len(relevant_profiles), 2))
        counter = 0
        for mp_idx, key in enumerate(relevant_mp_columns):
            if key in ['k', 'q95']:
                relevant_mp_vals = torch.zeros(len(relevant_profiles))
                mp_raw_data, mp_raw_time = mp_data[key]['data'], mp_data[key]['time']
                if mp_raw_time is None or isinstance(mp_raw_time, str):
                    pass
                else:
                    f = interp1d(mp_raw_time, mp_raw_data)
                    relevant_mp_vals = torch.from_numpy(f(relevant_time_windows))
                checkparams[:,counter] = relevant_mp_vals
                counter += 1
        # Check that q95 is lower than 12 and kappa is positive
        relevant_time_windows_bool_cor: torch.Tensor = torch.logical_and(torch.abs(checkparams[:,0] - 1.5) < 0.5, torch.abs(torch.abs(checkparams[:,1]) - 7.1) < 4.9)
        relevant_time_windows = relevant_time_windows[relevant_time_windows_bool_cor]
        relevant_profiles = relevant_profiles[relevant_time_windows_bool_cor]
        relevant_radii = relevant_radii[relevant_time_windows_bool_cor]
        #relevant_time_windows_bool: torch.Tensor = torch.logical_and(relevant_time_windows_bool1, relevant_time_windows_bool_cor)

    if limitmaxslices:
        if len(relevant_profiles) > 10:
            select_slices1 = torch.randperm(len(relevant_profiles))[:10]
            #select_slices = torch.randint(0, len(relevant_profiles), (10,))
            pressures = relevant_profiles[:,0]*relevant_profiles[:,1]*1.6022e-19
            index = torch.argmin(torch.abs(relevant_radii - 0.85),dim=1)
            index = torch.mean(index.float())
            index = index.long()
            pressure_slices = pressures[:,index]
            topvalues = torch.topk(pressure_slices,3)
            select_slices2 = topvalues.indices
            slicestemp = torch.cat((select_slices1, select_slices2))
            select_slices = torch.unique(slicestemp)
            #print('Entering te limit')
            select_slicesi = torch.where(relevant_profiles[select_slices,1,0] > 2.0e3, True, False)
            if sum(select_slicesi==True) == 0:
                print('shot ' + str(shot_num) +' no slices above 2 keV')
                return [None]*4
            #print('selecting slices')
            select_slices = select_slices[select_slicesi]
            #print('Leaving te limit')
            #if max_pressure:
            #    pressures = relevant_profiles[:,0]*relevant_profiles[:,1]*1.6022e-19
            #    index = torch.argmin(torch.abs(relevant_radii - 0.85),dim=1)
            #    index = torch.mean(index.float())
            #    index = index.long()
            #    print('taking pressure slices')
            #    pressure_slices = pressures[:,index]
            #    if len(pressure_slices) > 10:
            #        topvalues = torch.topk(pressure_slices,3)
            #        select_slices = topvalues.indices
            #    else:
            #        select_slices = torch.zeros(1)
            #else:        
            #    select_slices = torch.randint(0, len(relevant_profiles), (10,))
            print(select_slices)
            relevant_time_windows = relevant_time_windows[select_slices]
            relevant_profiles = relevant_profiles[select_slices]
            relevant_radii = relevant_radii[select_slices]

    relevant_machine_parameters: torch.Tensor = torch.empty((len(relevant_profiles),
                                                             len(relevant_mp_columns) + \
                                                             len(additional_feature_engineering_columns)))
    print('Looping mps')
    for mp_idx, key in enumerate(relevant_mp_columns): 
        relevant_mp_vals = torch.zeros(len(relevant_profiles))
        mp_raw_data, mp_raw_time = mp_data[key]['data'], mp_data[key]['time']
        if key == 'q95':
            mp_raw_data = np.abs(mp_raw_data)
        # this catches whenever NBI isn't working or the string in JET pulse files 'NO_ICRH_USED'
        if mp_raw_time is None or isinstance(mp_raw_time, str): 
            pass 
        else:
            f = interp1d(mp_raw_time, mp_raw_data)
            relevant_mp_vals = torch.from_numpy(f(relevant_time_windows))
        relevant_machine_parameters[:, mp_idx] = relevant_mp_vals

    for n_idx, feature in enumerate(additional_feature_engineering_columns, start=mp_idx+1): 
        if feature == 'aspect_ratio': 
            # major radius / minor radius 
            r_idx, a_idx = relevant_mp_columns.index('Rgeo'), relevant_mp_columns.index('ahor')
            relevant_machine_parameters[:, n_idx] = relevant_machine_parameters[:, r_idx] / relevant_machine_parameters[:, a_idx]
        elif feature == 'inverse_aspect_ratio': 
            # minor radius / major radius
            r_idx, a_idx = relevant_mp_columns.index('Rgeo'), relevant_mp_columns.index('ahor')
            relevant_machine_parameters[:, n_idx] = relevant_machine_parameters[:, a_idx] / relevant_machine_parameters[:, r_idx]
        elif feature == 'P_TOT':
            icr_idx, ecr_idx, oh_idx = relevant_mp_columns.index('PICR_TOT'), relevant_mp_columns.index('PECR_TOT'), \
                relevant_mp_columns.index('P_OH')
            if not mp_data.get('PNBI_TOT'):
                raise ValueError
            nbi_idx = relevant_mp_columns.index('PNBI_TOT')
            PECR_TOT = relevant_machine_parameters[:, ecr_idx]
            PECR_TOT = torch.where(PECR_TOT > 2e7, 2e7, PECR_TOT)
            relevant_machine_parameters[:,ecr_idx]=PECR_TOT
            PTOT = relevant_machine_parameters[:, nbi_idx] +\
                relevant_machine_parameters[:, icr_idx]+relevant_machine_parameters[:, ecr_idx] +\
                relevant_machine_parameters[:, oh_idx]
            PTOT = torch.where(PTOT>4e7, 4e7, PTOT)
            relevant_machine_parameters[:, n_idx] = PTOT
        elif feature == 'P_TOT/P_LH':
            power_scaling = lambda n, r, a, b : (2.14)*(np.exp(0.107))*(np.power(n*1e-20,0.728))*(np.power(abs(b),0.772))*(np.power(a,0.975))*(np.power(r,0.999))
            n = relevant_profiles[:,0,0]
            r = relevant_machine_parameters[:,relevant_mp_columns.index('Rgeo')]
            a = relevant_machine_parameters[:,relevant_mp_columns.index('ahor')]
            b = relevant_machine_parameters[:,relevant_mp_columns.index('BTF')]
            pmartin = power_scaling(n, r, a, b)
            nbi_idx, icr_idx, ecr_idx, oh_idx = relevant_mp_columns.index('PNBI_TOT'), \
                relevant_mp_columns.index('PICR_TOT'), relevant_mp_columns.index('PECR_TOT'), \
                relevant_mp_columns.index('P_OH')
            PTOT = relevant_machine_parameters[:, nbi_idx] +\
                relevant_machine_parameters[:, icr_idx]+relevant_machine_parameters[:, ecr_idx] +\
                relevant_machine_parameters[:, oh_idx]
            PTOT_PLH = PTOT/(1e6*pmartin)
            PTOT_PLH = torch.where(PTOT_PLH > 30, 30.0, PTOT_PLH)
            relevant_machine_parameters[:, n_idx] = PTOT_PLH
        elif feature in ['ped_col', 'ped_rhos', 'ped_beta']:
            pass
        else: 
            raise NotImplementedError(f'Feature transformation not defined for {feature}')

    print('Entering mapping functions')
    for mapping in mapping_functions:
        try:
            relevant_profiles, relevant_radii, relevant_machine_parameters, relevant_time_windows = globals()[mapping](relevant_profiles, relevant_radii, relevant_machine_parameters, relevant_time_windows)
        except KeyError as e: 
            raise NotImplementedError(f'{mapping}: does not exist as a mapping, please write it or remove it')
        except AttributeError: 
            return [None]*4
        else: 
            if any(v is None for v in [relevant_profiles, relevant_radii, relevant_machine_parameters, relevant_time_windows]): 
                return [None]*4
    print('Entering nondim')
    ped_col, ped_rhos, ped_beta = transform_to_nondimensional(relevant_profiles, relevant_radii,
                                                              relevant_machine_parameters, relevant_time_windows)
    for n_idx, feature in enumerate(additional_feature_engineering_columns, start=mp_idx+1): 
        if feature == 'ped_col':
            relevant_machine_parameters[:, -3] = ped_col
        if feature == 'ped_rhos':
            relevant_machine_parameters[:, -2] = ped_rhos
        if feature == 'ped_beta':
            relevant_machine_parameters[:, -1] = ped_beta
            
    if any(v.shape[0] == 0 for v in [relevant_profiles, relevant_radii, relevant_machine_parameters, relevant_time_windows]): 
        relevant_profiles, relevant_radii, relevant_machine_parameters, relevant_time_windows = [None]*4
    return relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_time_windows

from scipy.interpolate import interp1d
from scipy.constants import e as eV

def transform_to_nondimensional(profiles:torch.Tensor, radii: torch.Tensor,
                                mps, time, radindex=8) -> List[Union[torch.Tensor, np.ndarray]]: 
    def torch_shaping_approx(minor_radius, tri_u, tri_l, elongation):
        triangularity = (tri_u + tri_l) / 2.0
        b = elongation*minor_radius
        gamma_top = -(minor_radius + triangularity)
        gamma_bot = minor_radius - triangularity
        alpha_top = -gamma_top / (b*b)
        alpha_bot = -gamma_bot / (b*b)
        top_int = (torch.arcsinh(2*torch.abs(alpha_top)*b) + \
                   2*torch.abs(alpha_top)*b*torch.sqrt(4*alpha_top*alpha_top*b*b+1)) / (2*torch.abs(alpha_top))
        bot_int = (torch.arcsinh(2*torch.abs(alpha_bot)*b) + \
                   2*torch.abs(alpha_bot)*b*torch.sqrt(4*alpha_bot*alpha_bot*b*b+1)) / (2*torch.abs(alpha_bot))
        return bot_int + top_int 
    def bpol_approx(minor_radius, tri_u, tri_l, elongation, current): 
        shaping = torch_shaping_approx(minor_radius, tri_u, tri_l, elongation)
        return mu_0*current / shaping
    # Load mps that are needed
    r_idx, triu_idx, tril_idx, elo_idx = relevant_mp_columns.index('ahor'), \
                relevant_mp_columns.index('delRoben'), relevant_mp_columns.index('delRuntn'), \
                relevant_mp_columns.index('k')
    Ip_idx = relevant_mp_columns.index('IpiFP')
    BT_idx = relevant_mp_columns.index('BTF')
    q95_idx = relevant_mp_columns.index('q95')
    Rgeo_idx = relevant_mp_columns.index('Rgeo')
    rmin = torch.from_numpy(mps[:,r_idx])
    Rmaj = torch.from_numpy(mps[:,Rgeo_idx])
    triu = torch.from_numpy(mps[:,triu_idx])
    tril = torch.from_numpy(mps[:,tril_idx])
    elo = torch.from_numpy(mps[:,elo_idx])
    Ip = torch.from_numpy(mps[:,Ip_idx])
    Bt = torch.from_numpy(mps[:,BT_idx])
    q95 = torch.from_numpy(mps[:,q95_idx])
    inverse_aspect = rmin/Rmaj
    profiles = torch.from_numpy(profiles)
    profiles = profiles[:,:,radindex]
    # Calculate pressure to beta_p transformation
    bol = bpol_approx(rmin, triu, tril, elo, Ip)
    bolpres = bol*bol/(2*mu_0)
    # factor of 2 is here to approximate p_i = p_e
    betapres = 2*profiles[:,-1]/bolpres 
    # Calculate temperature to rho* transformation
    intermvalue1 = (torch.abs(Bt)*rmin)
    intermvalue = 1.0/intermvalue1
    prefactor = intermvalue*torch.sqrt(torch.tensor(2*2*m_p/e))
    rhostar = prefactor*torch.sqrt(profiles[:,1])
    # Calculate density to collisionality transformation
    coulomb_log = 31.3 - torch.log(torch.sqrt(profiles[:,0])/profiles[:,1])
    prefactor1 = 6.921*1e-18*coulomb_log
    q95abs = torch.abs(q95)
    term11 = (Rmaj*q95abs)
    term1 = term11*profiles[:,0]
    invpov1 = torch.pow(inverse_aspect,1.5)
    invpov = invpov1
    profpov = torch.pow(profiles[:,1],2)
    term2 = invpov*profpov
    collisionality = prefactor1*term1/term2
    #6.921*1e-18*coulomb_log*((Rgeo*torch.abs(q95)).view(-1,1)*profiles[:,0]/(torch.power(inserve_aspect,1.5).view(-1,1)*torch.power(profiles[:,1],2)))
    #prof_remap = np.stack([collisionality, rhostar, betapres], axis=1)
    ped_col = torch.where(collisionality>10.0, 10.0, collisionality)
    ped_col = torch.where(ped_col < 0.01, 0.01, ped_col)
    torch.nan_to_num(ped_col, nan=10.0, posinf=10.0, neginf=0.01)
    ped_col_list.append(ped_col)
    ped_rhos = rhostar
    ped_beta = betapres
    return ped_col, ped_rhos, ped_beta
    #return prof_remap, radii, mps, time

def example_mapping(profiles:torch.Tensor, radii: torch.Tensor,
                    mps, time, map_JET_sep = True,
                    fix_low_te = True) -> List[Union[torch.Tensor, np.ndarray]]: 
    profiles_remapped, radii_remapped, mps_remapped, times_remapped = [], [], [], []
    psi_lb, psi_ub, num_points = 0.85, 1.05, 30
    remapx_axis = np.linspace(psi_lb, psi_ub, num_points)
    if 'AUG' in folder_name: 
        radii = radii**2 # AUG DATA comes in RHO
    removed_slices = 0
    for slice_idx in range(profiles.shape[0]): 
        # if radii[slice_idx].min() > psi_lb and radii[slice_idx].max() < psi_ub: 
        #     print('Some slices fall out of the psi interp window, skipping those slices')
        try:
            radval = radii[slice_idx]
            teval = profiles[slice_idx, 1]
            neval =  profiles[slice_idx, 0]
            if map_JET_sep:
                if fix_low_te:
                    neval = profiles[slice_idx, 0]
                    ind = np.where(neval < 5e18)
                    ind1 = np.where(teval > 50)
                    index = np.intersect1d(ind,ind1)
                    teval[index] = neval[index]*50/5e18
                    ind = np.where(neval<1e19)
                    ind1 = np.where(teval*neval*1.6022e-19 > 400)
                    index1 = np.intersect1d(ind,ind1)
                    teval[index1] = neval[index1]*100/1e19
                f_te = interp1d(radval, teval)
                xrtest = np.linspace(0.85,1.05,1000)
                tevs = f_te(xrtest)
                index1 = np.where(tevs - 100 < 0)
                psinsep = xrtest[index1[0][0]]
                radval = radval + (1.0 - psinsep)
            f_ne = interp1d(radval, neval)
            ne_remap = f_ne(remapx_axis)
            f_te = interp1d(radval, teval)
            te_remap = f_te(remapx_axis)
            if fix_low_te:
                ind1 = np.where(remapx_axis > 1.0)
                ind2 = np.where(te_remap > 150)
                index1 = np.intersect1d(ind1,ind2)
                te_remap[index1] = 100*np.exp(-(remapx_axis[index1] - 1.0)/0.03) 
           # pe_remap = eV*np.multiply(ne_remap, te_remap)
           # prof_remap = np.stack([ne_remap, te_remap, pe_remap], axis=0)
            prof_remap = np.stack([ne_remap, te_remap], axis=0)
            profiles_remapped.append(prof_remap)
            radii_remapped.append(remapx_axis)
            mps_remapped.append(mps[slice_idx])
            times_remapped.append(time[slice_idx])

        except ValueError: 
            # shot_num = shot.split('/')[-1]
            # print(f'Slice {slice_idx} in {shot_num} (time = {time[slice_idx]}) falls outside of the interpolation range of {psi_lb}, {psi_ub}: MAX {radii[slice_idx].max():.4}, MIN {radii[slice_idx].min():.5}')
            removed_slices += 1
            continue 

    print(f'{removed_slices} slices removed of original {len(profiles)}, ')
    if len(profiles_remapped) == 0: 
        return [None]*4
    else: 
        return np.stack(profiles_remapped,0), np.stack(radii_remapped,0), np.stack(mps_remapped,0), np.stack(times_remapped,0)
    

def save_shot_arrays(save_dir: str, final_profiles:Union[np.ndarray, torch.Tensor], final_mps: Union[np.ndarray, torch.Tensor], final_radii: Union[np.ndarray, torch.Tensor], final_times: Union[np.ndarray, torch.Tensor], shot_num: int) -> None: 
    relevant_path = os.path.join(save_dir, shot_num)
    with open(relevant_path + '_MP.npy', 'wb') as mp_file: 
        if isinstance(final_mps, torch.Tensor):
            final_mps = final_mps.numpy() 
        np.save(mp_file, final_mps)
    with open(relevant_path + '_PROFS.npy', 'wb') as prof_file: 
        if isinstance(final_profiles, torch.Tensor):
            final_profiles = final_profiles.numpy() 
        np.save(prof_file, final_profiles)
    with open(relevant_path + '_RADII.npy', 'wb') as radii_file: 
        if isinstance(final_radii, torch.Tensor):
            final_radii = final_radii.numpy() 
        np.save(radii_file, final_radii)
    with open(relevant_path + '_TIME.npy', 'wb') as time_file: 
        if isinstance(final_times, torch.Tensor):
            final_times = final_times.numpy() 
        np.save(time_file, final_times)

    print(f'Saved to {relevant_path}\n')
def build(shot: str): 
    shot_num = shot.split('/')[-1]
    try:
        print('Entering convert raw file to tensor')
        profiles, mps, radii, times = convert_raw_file_to_tensor(shot, relevant_mp_columns,
                                                                 additional_feature_engineering_cols,
                                                                 mapping_funcs)
        print('Leaving convert raw file to tensor')
    except ValueError as e: 
        print(e, shot_num)
        profiles = None 
    except IndexError as e: 
        print(e, shot_num)
        profiles = None 
    except pickle.UnpicklingError as e: 
        print(e, shot_num)
        profiles = None 
    if profiles is not None: 
        assert profiles.shape[0] == mps.shape[0]
        assert radii.shape[0] == times.shape[0]
        save_shot_arrays(SAVE_DIR, profiles, mps, radii, times, shot_num)
    

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Process raw data into numpy arrays')
    parser.add_argument('-rf', '--raw_folder_name', action='append', type=str, required=True)
    parser.add_argument('-lf', '--array_folder_name', type=str, required=True)
    parser.add_argument('-feature', '--additional_feature_engineering', action='append', type=str, default=[])
    parser.add_argument('-mf', '--additional_mapping_functions', action='append', default=[], type=str)
    parser.add_argument('-mp', action='count', default=0, help='To do multiprocessing or not')
    args = parser.parse_args()
    
    relevant_mp_columns = ['BTF', 'IpiFP', 'D_tot', 'PNBI_TOT', 'PICR_TOT','PECR_TOT', 'P_OH', 'k', 'delRoben', 'delRuntn', 'ahor', 'Rgeo', 'q95', 'Vol']
    additional_feature_engineering_cols = args.additional_feature_engineering
    mapping_funcs = args.additional_mapping_functions
    jet_pdb = pd.read_csv('/scratch/project_2005083/aarojarvinen/ped_ssm/jet-all-full.csv')
    # for debugging pedestal_collisionalities
    ped_col_list = []
    
    SAVE_DIR = args.array_folder_name
    print('MAKING TRANSFORMATIONS ON THE FOLLOWING RAW DIRS', args.raw_folder_name)
    for folder_name in args.raw_folder_name: 
        RAW_AUG_DIR = folder_name
        shot_files = sorted([os.path.join(RAW_AUG_DIR, file) for file in os.listdir(RAW_AUG_DIR)])
        print(len(shot_files))
        if args.mp: 
            print(f'Doing multiprocessing with {psutil.cpu_count(logical=False)} cpus')
            with Pool(psutil.cpu_count(logical=False)) as pool: 
                pool.map(build, shot_files)
        else: 
            print('going one by one ')
            for shot in shot_files: 
                build(shot)

    # for debugging pedestal_collisionalities
    peds = np.array(ped_col_list)
    np.savetxt('pedesta_col_debug.txt',peds)
    
    print('FINISHED ALL TRANSOFMRATIONS, NOW SAVING MPS')
    with open(os.path.join(SAVE_DIR, 'mp_names_saved.txt'), 'w') as f:
        line = ','.join(relevant_mp_columns + additional_feature_engineering_cols)
        # for line in relevant_mp_columns:
        f.write(f"{line}")
    print(f'Saved mp names to {SAVE_DIR}mp_names_saved.txt')

    # already_saved = sorted([shot_num for shot_num in os.listdir(RAW_AUG_DIR) if (os.path.exists(os.path.join(SAVE_DIR, f'{shot_num}_PROFS.npy'))) & (os.path.exists(os.path.join(SAVE_DIR, f'{shot_num}_MP.npy'))) & (os.path.exists(os.path.join(SAVE_DIR, f'{shot_num}_RADII.npy'))) & (os.path.exists(os.path.join(SAVE_DIR, f'{shot_num}_TIME.npy')))])
    
# -rf /home/kitadam/ENR_Sven/moxie/data/raw/RAW_AUG_PULSES -rf /home/kitadam/ENR_Sven/moxie/data/raw/RAW_JET_PULSES -lf /home/kitadam/ENR_Sven/ped_ssm/local_trial_jet_aug
