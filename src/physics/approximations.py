import numpy as np 
from typing import List
from sklearn.linear_model import LinearRegression 
import ruptures

mu_0 = 4 * np.pi * 10 ** -7 # magnetic permability 
m_i = 3.3435860e-27  # Mass of an ion of deuterium (kg) kg
m_p = 1.6726219e-27  # Mass of a proton (kg) 
m_e = 9.10938356e-31  # Mass of an electron (kg)




def find_max_value_of_array_near_separatrix(arr: np.ndarray, slice_radius: np.ndarray): 
    in_ped_mask = np.logical_and(slice_radius > 0.6, slice_radius < 1.1)
    arr_in_ped, slice_radius_in_ped = arr[in_ped_mask], slice_radius[in_ped_mask]
    if np.isnan(arr_in_ped).sum() > 0: # TODO: THIS IS HACK SHOULD FIX SO THAT NO NANS ARE IN THE BOOSTRAP CURRENT
        arr_in_ped = np.nan_to_num(arr_in_ped, nan = -500000)
    
    max_idx = np.argmax(arr_in_ped)
    max_val_in_arr_in_ped = arr_in_ped[max_idx]
    radius_of_max_val_in_arr_in_ped = slice_radius_in_ped[max_idx]
    idx_of_max_location_in_original_array = np.where(slice_radius == radius_of_max_val_in_arr_in_ped)[0][0]
    return max_val_in_arr_in_ped, radius_of_max_val_in_arr_in_ped, idx_of_max_location_in_original_array

def calculate_stability_profiles_and_maxes_for_pulse(profiles: np.ndarray, mps: np.ndarray, radii: np.ndarray, times: np.ndarray, mp_names: List[str]) -> List[np.ndarray]: 
    volume_idx = mp_names.index('Vol')
    radius_idx = mp_names.index('Rgeo')
    vol = mps[:, volume_idx]
    major_radius = mps[:, radius_idx]
    psi = radii[0]*radii[0]  # we take advantage that rho is the same for all time slices in discharge, shoutout to Rainer for that
    k_boltzmann = 1.380649e-23  # Boltzmann constant in m^2 kg s^-2 K^-1
    ev_to_k = 11604.505  # Conversion factor from eV to K
    pe = np.prod(profiles, 1)*ev_to_k*k_boltzmann
    alpha_normalized_pressure_gradient = normalized_pressure_gradient_approximation(pe, psi, major_radius, vol)

    ne, te = profiles[:, 0], profiles[:, 1]
    minor_radius = mps[:, mp_names.index('ahor')]
    total_plasma_current = mps[:, mp_names.index('IpiFP')]
    q_95 = mps[:, mp_names.index('q95')]

    profiles_all = [ne, te, pe]
    
    bootstrap = bootstrap_current_redl(profiles_all, psi, q_95, total_plasma_current, major_radius, minor_radius)

    jb_maxes, alpha_maxes =  np.zeros_like(times), np.zeros_like(times)
    for idx, t in enumerate(times): 
        max_jb, _, _ = find_max_value_of_array_near_separatrix(bootstrap[idx] / np.mean(bootstrap[idx]), psi)
        max_alpha, _, _ = find_max_value_of_array_near_separatrix(alpha_normalized_pressure_gradient[idx], psi)
        jb_maxes[idx] = max_jb
        alpha_maxes[idx] = max_alpha

    jb_maxes = np.clip(jb_maxes, a_min=0.001, a_max=10.0)
    alpha_maxes = np.clip(alpha_maxes, a_min=0.001, a_max=10.0)
    thetas = np.arctan(jb_maxes / alpha_maxes) *180 / np.pi

    norm_mag = np.sqrt(alpha_maxes*alpha_maxes + jb_maxes*jb_maxes)
    norm_mag = np.clip(norm_mag, a_min=0.001, a_max=30.0)
    return alpha_maxes, jb_maxes, norm_mag, thetas, alpha_normalized_pressure_gradient, bootstrap

def calculate_stability_labels(thetas: np.ndarray, norm_mag: np.ndarray, alpha_maxes: np.ndarray, jb_maxes: np.ndarray, plasma_current: np.ndarray, plh_ratio: np.ndarray, time:np.array): 
    def standardize(arr: np.ndarray): 
        return (arr - arr.mean()) / np.clip(arr.std(), a_min=0.1, a_max=None)
    algo = ruptures.Binseg(min_size=20).fit(np.stack([standardize(arr) for arr in [thetas, norm_mag, alpha_maxes, jb_maxes, plh_ratio, plasma_current]], 1))
    results = algo.predict(pen=np.log(len(jb_maxes)*6))
    results.insert(0, 0)
    labels = np.zeros_like(norm_mag)
    ratios = np.zeros_like(norm_mag)
    mean_theta, std_theta = np.mean(thetas), np.std(thetas)
    for i in range(len(results) -1): 
        beg, end = results[i], results[i+1]
        if end == len(norm_mag): 
            end = end - 1
        t_window_bool = np.logical_and(time > time[beg], time < time[end])
        t_window = time[t_window_bool]
        normed_thetas = (thetas[t_window_bool] - np.mean(thetas[t_window_bool])) / np.std(thetas[t_window_bool])
        normed_norm_mag =  (norm_mag[t_window_bool] - np.mean(norm_mag[t_window_bool])) / np.std(norm_mag[t_window_bool])
        model_theta = LinearRegression()
        model_theta.fit(t_window.reshape(-1, 1), normed_thetas.reshape(-1, 1))

        model_norm = LinearRegression()
        model_norm.fit(t_window.reshape(-1, 1), normed_norm_mag.reshape(-1, 1))
        if abs(model_theta.coef_) < 1.0 and np.mean(thetas[t_window_bool]) < mean_theta + std_theta and np.mean(thetas[t_window_bool]) > mean_theta - std_theta and abs(model_norm.coef_) < 1.0: 
            window_norm_mag = norm_mag[t_window_bool]
            upper_90_mag = np.quantile(window_norm_mag, 0.9)
            idx_bool_of_boundary_from_upper_quantile = window_norm_mag > upper_90_mag
            theta_values_at_pseudo_boudnary = thetas[t_window_bool][idx_bool_of_boundary_from_upper_quantile]
            theta_mean = np.mean(theta_values_at_pseudo_boudnary)        
            color = None
            if theta_mean >= 40 and theta_mean <= 50: 
                label = 'P-B'
                label_num = 3
            elif theta_mean > 50 and theta_mean <= 90:
                label = 'P'
                label_num = 2
            elif theta_mean > 0 and theta_mean < 40:
                label = 'B'
                label_num = 1
            else: 
                label = 'T'
                label_num = 0
        else: 
            label = 'T'
            label_num = 0

        if label_num != 0: 
            # calculate the ratio between values within and the norm mag itself
            max_in_window = np.max(norm_mag[t_window_bool])
            ratio = norm_mag[beg:end] / max_in_window
        else: 
            ratio = np.zeros_like(labels[beg:end])

        ratios[beg:end] = ratio
        labels[beg:end] = np.ones_like(labels[beg:end])*label_num
    return labels, ratios

def volume_approximation(volume: np.ndarray, psi:np.ndarray) -> np.ndarray:
    psi = psi.reshape((1, 200))
    scaled_volume = np.where(psi < 1.0, psi**4 * volume[:, np.newaxis], volume[:, np.newaxis])
    return scaled_volume


def q_profile_approximation(q_95: np.ndarray, psi: np.ndarray, n: int=3, a: float=1.1)  -> np.ndarray: 
    b = ((abs(q_95)[:, np.newaxis] / 1.1)**(1.0 / (0.95))**n)
    q_psi = lambda x: a*(b**(x**n))
    q_profile = q_psi(psi)    
    return q_profile

def normalized_pressure_gradient_approximation(pressure: np.ndarray, psi:np.ndarray, major_radius: np.ndarray, total_volume: np.ndarray)  -> np.ndarray: 
    V_psi = volume_approximation(total_volume, psi)
    c1 = np.gradient(V_psi, psi, axis=-1) # *2 / ((2*np.pi)**2)
    c2 = (V_psi / (2*np.pi*np.pi*np.clip(major_radius[:, np.newaxis], a_min=0.001, a_max=None)))**(1/2) # clipping due to zero values
    grad_pressure = np.gradient(pressure, psi, axis=-1)
    alpha = -c1*c2*grad_pressure*mu_0
    return alpha 


# correct 
def calculate_L31(f_trap, Z_eff, v_e_star):
    X31 = calculate_f_t31_eff(f_trap, Z_eff, v_e_star)

    term_denominator = (Z_eff**1.2 - 0.71)
    term1 =  (1 + 0.15 / term_denominator) * X31
    term2 = -((0.22 / term_denominator) * X31**2)
    term3 =  (0.01 / term_denominator) * X31**3
    term4 =  (0.06 / term_denominator) * X31**4

    L31 = term1 + term2 + term3 + term4
    return L31

# correct 
def calculate_f_t31_eff(f_trap, Z_eff, v_e_star):
    numerator = f_trap
    denominator_term_2 = (0.67 * (1 - 0.7 * f_trap) * v_e_star**0.5) / (0.56 + 0.44 * Z_eff)
    denominator_term_3 = (0.52 + 0.086 * v_e_star**0.5 * (1 + 0.87 * f_trap) * v_e_star) / (1 + 1.13 * (Z_eff - 1)**0.5)
    denominator = 1 + denominator_term_2  + denominator_term_3
    f_t31_eff = numerator / denominator
    return f_t31_eff

def calculate_L32(f_trap, Z_eff, v_e_star): 
    F_32_ei = calculate_F32_ei(f_trap, Z_eff, v_e_star)
    F_32_ee = calculate_F32_ee(f_trap, Z_eff, v_e_star)
    return F_32_ee + F_32_ei

def calculate_F32_ee(f_trap, Z_eff, v_e_star):
    X32_e = calculate_f_t32_ee_eff(f_trap, Z_eff, v_e_star)
    
    term1_numerator = 0.1 + 0.6 * Z_eff
    term1_denominator = Z_eff * (0.77 + 0.63 * (1 + (Z_eff - 1)**1.1))
    term1 = (term1_numerator / term1_denominator) * (X32_e - X32_e**4)

    term2_numerator = 0.7
    term2_denominator = (1 + 0.2 * Z_eff)
    term2 = (term2_numerator / term2_denominator) * (X32_e**2 - X32_e**4 - 1.2 * (X32_e**3 - X32_e**4))
    term3 = (1.3 / (1 + 0.5 * Z_eff)) * X32_e**4

    F32_ee = term1 + term2 + term3
    return F32_ee

def calculate_f_t32_ee_eff(f_trap, Z_eff, v_e_star): 
    numerator = f_trap 
    denominator_1 = 1
    denominator_2 =  (0.23*(1 - 0.96*f_trap)*np.sqrt(v_e_star)) / (np.sqrt(Z_eff))
    denominator_3_mult = (0.13*(1-0.38*f_trap)*v_e_star) / (Z_eff**0.5)
    denominator_3 = denominator_3_mult*( np.sqrt(1+2*np.sqrt(Z_eff - 1)) + (f_trap**2)*np.sqrt( (0.075 + 0.25* (Z_eff - 1)**2)*v_e_star))
    return numerator / (denominator_1 + denominator_2 + denominator_3)

def calculate_F32_ei(f_trap, Z_eff, v_e_star):
    X_32_ei = calculate_f_t32_ei_eff(f_trap, Z_eff, v_e_star)

    term_1 = ((0.4 + 1.93*Z_eff) / ( Z_eff*(0.8 + 0.6*Z_eff))) * (X_32_ei - X_32_ei**4)
    term_2 = (5.5 / (1.5 + 2*Z_eff)) * (X_32_ei**2 - X_32_ei**4 - 0.8*(X_32_ei**3 - X_32_ei**4))
    term_3 = (1.3 / (1+0.5*Z_eff))* (X_32_ei**4)
    F_32_ei = -term_1 + term_2 -term_3 
    return F_32_ei 

def calculate_f_t32_ei_eff(f_trap, Z_eff, v_e_star):
    f_t32_ei_eff = f_trap / (1 + ((0.87 * (1 + 0.39 * f_trap) * np.sqrt(v_e_star)) / (1 + 2.95 * (Z_eff - 1)**2)) + 1.53 * (1 - 0.37 * f_trap) * v_e_star * (2 + 0.375 * (Z_eff - 1)))
    return f_t32_ei_eff


def calculate_sigma_neo_over_sigma_Spitzer(f_trap, Z_eff, v_e_star):
    X33 = calculate_f_t33_eff(f_trap, Z_eff, v_e_star)
    sigma_neo_over_sigma_Spitzer = 1 - (1 + 0.21 / Z_eff) * X33 + (0.54 / Z_eff) * X33**2 - (0.33 / Z_eff) * X33**3
    return sigma_neo_over_sigma_Spitzer


def calculate_f_t33_eff(f_trap, Z_eff, v_e_star):
    f_t33_eff = f_trap / (1 + 0.25 * (1 - 0.7 * f_trap) * np.sqrt(v_e_star) * (1 + 0.45 * (Z_eff - 1)**0.5) + (0.61 * (1 - 0.41 * f_trap) * v_e_star) / (Z_eff**0.5))
    return f_t33_eff


def calculate_alpha_0(f_trap, Z_eff):
    alpha_0 = -(((0.62 + 0.055 * (Z_eff - 1)) / (0.53 + 0.17 * (Z_eff - 1))) * ( (1 - f_trap) / (1 - (0.31 - 0.065 * (Z_eff - 1)) * f_trap - 0.25 * f_trap**2)))
    return alpha_0


def calculate_alpha_jb(f_trap, Z_eff, v_i_star):
    alpha_0 = calculate_alpha_0(f_trap, Z_eff)
    alpha = ((alpha_0 + 0.7 * Z_eff * np.sqrt(v_i_star)*np.sqrt(f_trap)) / (1 + 0.18 * np.sqrt(v_i_star) - 0.002 * v_i_star**2 * f_trap**6)) * (1 / (1 + 0.004 * v_i_star**2 * f_trap**6))
    return alpha

def bootstrap_current_redl(profiles_all: List[np.ndarray], psi: np.ndarray, q_95: np.ndarray, total_plasma_current: np.ndarray, major_radius: np.ndarray, minor_radius: np.ndarray) -> np.ndarray: 
    ne, te, pe = profiles_all
    q_profile = q_profile_approximation(q_95, psi)

    current_profile = total_plasma_current[:, np.newaxis]*np.ones_like(psi)

    epsilon = (minor_radius / major_radius)[:, np.newaxis] # inverse aspect ratio 
    f_trap = np.sqrt(2) * epsilon  # ratio of trapped to circ1ulating particles 
    Z_eff = 1.0 # also approximation! 

    p = 2*pe 
    pi = pe 
    n = 2*ne 
    ti = te 
    ni = ne 


    ln_cloumb_e = 31.3 - np.log(np.sqrt(ne) / te)
    ln_cloumb_ii = 30.0 - np.log((Z_eff**3 )* np.sqrt(ni) / np.power(ti, 1.5))

    nu_e_star = (6.921E-18)*(q_profile*major_radius[:, np.newaxis]*ne*Z_eff*ln_cloumb_e) / (te*te*(epsilon**(1.5)))
    nu_i_star = (4.9E-18)*(q_profile*major_radius[:, np.newaxis]*ni*(Z_eff**4)*ln_cloumb_ii) / (ti*ti*(epsilon**(1.5)))
    sigma_neo = calculate_sigma_neo_over_sigma_Spitzer(f_trap, Z_eff, nu_e_star)

    L_31 = calculate_L31(f_trap, Z_eff, nu_e_star)
    L_32 = calculate_L32(f_trap, Z_eff, nu_e_star)
    L_34 = L_31
    alpha_jb = calculate_alpha_jb(f_trap, Z_eff, nu_i_star)

    term_1 = p*L_31 * np.gradient(np.log(n), psi, axis=1)
    term_2 = pe*(L_31 + L_32) * (np.gradient(np.log(te), psi, axis=1))
    term_3 = pi*(L_31 + L_34*alpha_jb)*(np.gradient(np.log(ti), psi, axis=1))
    bootstrap = sigma_neo - current_profile*(term_1 + term_2 + term_3)
    return bootstrap