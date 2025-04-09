#import all the libraries needed
from import_dep import *


# --- Unified Model Function ---
def unified_circuit_model(params, frequency, model_type='RC', param_order=None, fixed_params_map=None):
    """
    Calculates impedance for various equivalent circuit models.

    Supported models:
    'RC':   R_s + p(R1, C1+Cpad)
    'CPE':  R_s + p(R1, CPE1+Cpad)
    'RC2':  R_s + p(R1, C1+Cpad) + p(R2, C2)        <-- Cpad only with first RC
    'CPE2': R_s + p(R1, CPE1+Cpad) + p(R2, CPE2)    <-- Cpad only with first R||CPE
    'RC3':  R_s + p( C_pad, (p(R1,C1) + p(R2,C2)) )
    'CPE3': R_s + p( C_pad, (p(R1,CPE1) + p(R2,CPE2)) )

    Args: [... same ...] Returns: [... same ...]
    """
    # Define parameter orders locally (R_s is LAST)
    PARAM_ORDERS = {
        'RC':   ['R_mem1', 'C_mem1', 'C_pad', 'R_series'],
        'CPE':  ['R_mem1', 'Q1', 'alpha1', 'C_pad', 'R_series'],
        'RC2':  ['R_mem1', 'C_mem1', 'R_mem2', 'C_mem2', 'C_pad', 'R_series'],
        # --- CORRECTED CPE2 PARAM ORDER ---
        'CPE2': ['R_mem1', 'Q1', 'alpha1', 'R_mem2', 'Q2', 'alpha2', 'C_pad', 'R_series'],
        'RC3':  ['R_mem1', 'C_mem1', 'R_mem2', 'C_mem2', 'C_pad', 'R_series'], # Same params as RC2
        'CPE3': ['R_mem1', 'Q1', 'alpha1', 'R_mem2', 'Q2', 'alpha2', 'C_pad', 'R_series'] # Same params as CPE2
    }
    if model_type not in PARAM_ORDERS: raise ValueError(f"Unknown model_type '{model_type}'.")

    current_param_order = PARAM_ORDERS[model_type]
    expected_len = len(current_param_order)
    full_params = None

    # --- Reconstruction Logic (same as before) ---
    if param_order is not None and fixed_params_map is not None:
        if param_order != current_param_order: raise ValueError("param_order mismatch")
        if len(param_order) != len(params) + len(fixed_params_map): raise ValueError("Length mismatch")
        full_params = [None] * len(param_order); free_param_idx = 0
        for i in range(len(param_order)):
            if i in fixed_params_map: full_params[i] = fixed_params_map[i]
            else:
                if free_param_idx >= len(params): raise IndexError("More free params expected.")
                full_params[i] = params[free_param_idx]; free_param_idx += 1
        if free_param_idx != len(params): raise ValueError("Did not use all free params.")
    else:
        full_params = list(params)
    if len(full_params) != expected_len: raise ValueError(f"Incorrect # params for '{model_type}'")
    # --- End Reconstruction ---

    omega = 2 * np.pi * frequency
    epsilon = 1e-18
    Z_parallel1 = 0; Z_parallel2 = 0; R_series_val = 0; Z_model = 0

    # Helper
    def calculate_parallel_Z(R_val, Y_elem_val, eps):
        R_safe = max(R_val, eps)
        with np.errstate(divide='ignore', invalid='ignore'): Y_R = 1 / R_safe; Z_inv = Y_R + Y_elem_val
        with np.errstate(divide='ignore', invalid='ignore'): Z_inv_safe = np.where(np.abs(Z_inv) < eps, eps, Z_inv); Z_p = 1 / Z_inv_safe
        return np.nan_to_num(Z_p, nan=np.inf)

    # --- Model Calculations ---
    if model_type == 'RC': # R_s + p(R1, C1+Cpad)
        R_mem1, C_mem1, C_pad, R_series = full_params
        C_par = max(C_mem1, 0) + max(C_pad, 0); Y_elem = 1j * omega * C_par
        Z_parallel1 = calculate_parallel_Z(R_mem1, Y_elem, epsilon)
        R_series_val = max(R_series, 0)
        Z_model = R_series_val + Z_parallel1

    elif model_type == 'CPE': # R_s + p(R1, CPE1+Cpad)
        R_mem1, Q1, alpha1, C_pad, R_series = full_params
        Q1_safe = max(Q1, epsilon); alpha1_safe = max(min(alpha1, 1.0), epsilon); C_pad_safe = max(C_pad, 0)
        Y_CPE = Q1_safe * (1j * omega)**alpha1_safe; Y_Cpad = 1j * omega * C_pad_safe
        Y_elem_Total = Y_CPE + Y_Cpad
        Z_parallel1 = calculate_parallel_Z(R_mem1, Y_elem_Total, epsilon)
        R_series_val = max(R_series, 0)
        Z_model = R_series_val + Z_parallel1

    elif model_type == 'RC2': # R_s + p(R1, C1+Cpad) + p(R2, C2)
        R_mem1, C_mem1, R_mem2, C_mem2, C_pad, R_series = full_params
        C_par1 = max(C_mem1, 0) + max(C_pad, 0); Y_elem1 = 1j * omega * C_par1
        Z_parallel1 = calculate_parallel_Z(R_mem1, Y_elem1, epsilon)
        C_par2 = max(C_mem2, 0); Y_elem2 = 1j * omega * C_par2
        Z_parallel2 = calculate_parallel_Z(R_mem2, Y_elem2, epsilon)
        R_series_val = max(R_series, 0)
        Z_model = R_series_val + Z_parallel1 + Z_parallel2

    # --- CORRECTED CPE2 Calculation ---
    elif model_type == 'CPE2': # R_s + p(R1, CPE1+Cpad) + p(R2, CPE2)
        R_mem1, Q1, alpha1, R_mem2, Q2, alpha2, C_pad, R_series = full_params # Added C_pad
        # First parallel element (R1 || (CPE1 + Cpad))
        Q1_safe = max(Q1, epsilon); alpha1_safe = max(min(alpha1, 1.0), epsilon); C_pad_safe = max(C_pad, 0)
        Y_CPE1 = Q1_safe * (1j * omega)**alpha1_safe; Y_Cpad = 1j * omega * C_pad_safe
        Y_elem1_Total = Y_CPE1 + Y_Cpad
        Z_parallel1 = calculate_parallel_Z(R_mem1, Y_elem1_Total, epsilon)
        # Second parallel element (R2 || CPE2)
        Q2_safe = max(Q2, epsilon); alpha2_safe = max(min(alpha2, 1.0), epsilon)
        Y_elem2 = Q2_safe * (1j * omega)**alpha2_safe
        Z_parallel2 = calculate_parallel_Z(R_mem2, Y_elem2, epsilon)
        R_series_val = max(R_series, 0)
        Z_model = R_series_val + Z_parallel1 + Z_parallel2

    elif model_type == 'RC3': # R_s + p( C_pad, (p(R1,C1) + p(R2,C2)) )
        R_mem1, C_mem1, R_mem2, C_mem2, C_pad, R_series = full_params
        C_par1 = max(C_mem1, 0); Y_elem1 = 1j * omega * C_par1
        Z_parallel1 = calculate_parallel_Z(R_mem1, Y_elem1, epsilon)
        C_par2 = max(C_mem2, 0); Y_elem2 = 1j * omega * C_par2
        Z_parallel2 = calculate_parallel_Z(R_mem2, Y_elem2, epsilon)
        Z_parallel_combo = Z_parallel1 + Z_parallel2
        C_pad_val = max(C_pad, 0); Z_intermediate = Z_parallel_combo
        if C_pad_val > epsilon:
             with np.errstate(divide='ignore', invalid='ignore'):
                 Y_Cpad = 1j * omega * C_pad_val
                 Z_combo_inv = 1 / np.where(np.abs(Z_parallel_combo)<epsilon, epsilon, Z_parallel_combo)
                 Z_int_inv = Z_combo_inv + Y_Cpad; Z_int_inv_safe = np.where(np.abs(Z_int_inv)<epsilon,epsilon,Z_int_inv)
                 Z_intermediate = 1 / Z_int_inv_safe; Z_intermediate = np.nan_to_num(Z_intermediate, nan=np.inf)
        R_series_val = max(R_series, 0)
        Z_model = R_series_val + Z_intermediate

    elif model_type == 'CPE3': # R_s + p( C_pad, (p(R1,CPE1) + p(R2,CPE2)) )
        R_mem1, Q1, alpha1, R_mem2, Q2, alpha2, C_pad, R_series = full_params
        Q1_safe = max(Q1, epsilon); alpha1_safe = max(min(alpha1, 1.0), epsilon)
        Y_elem1 = Q1_safe * (1j * omega)**alpha1_safe
        Z_parallel1 = calculate_parallel_Z(R_mem1, Y_elem1, epsilon)
        Q2_safe = max(Q2, epsilon); alpha2_safe = max(min(alpha2, 1.0), epsilon)
        Y_elem2 = Q2_safe * (1j * omega)**alpha2_safe
        Z_parallel2 = calculate_parallel_Z(R_mem2, Y_elem2, epsilon)
        Z_parallel_combo = Z_parallel1 + Z_parallel2
        C_pad_val = max(C_pad, 0); Z_intermediate = Z_parallel_combo
        if C_pad_val > epsilon:
             with np.errstate(divide='ignore', invalid='ignore'):
                 Y_Cpad = 1j * omega * C_pad_val
                 Z_combo_inv = 1 / np.where(np.abs(Z_parallel_combo)<epsilon, epsilon, Z_parallel_combo)
                 Z_int_inv = Z_combo_inv + Y_Cpad; Z_int_inv_safe = np.where(np.abs(Z_int_inv)<epsilon,epsilon,Z_int_inv)
                 Z_intermediate = 1 / Z_int_inv_safe; Z_intermediate = np.nan_to_num(Z_intermediate, nan=np.inf)
        R_series_val = max(R_series, 0)
        Z_model = R_series_val + Z_intermediate

    return Z_model

# --- Weighted Residual & Cost Functions (No change needed) ---
def residuals_ls_weighted(params, frequency, Z_measured, model_type, param_order, fixed_params_map):
    # ... (same as before) ...
    Z_model = unified_circuit_model(params, frequency, model_type, param_order, fixed_params_map)
    Z_diff = Z_measured - Z_model; eps_w = 1e-9; mod_Z = np.abs(Z_measured)
    weights = 1. / np.maximum(mod_Z, eps_w*np.mean(mod_Z)+eps_w)
    return np.concatenate((Z_diff.real*weights, Z_diff.imag*weights))

def cost_function_de(params, frequency, Z_measured, model_type, param_order, fixed_params_map):
    # ... (same as before) ...
    return np.sum(residuals_ls_weighted(params, frequency, Z_measured, model_type, param_order, fixed_params_map)**2)


# --- Main Fitting Function ---
def fit_impedance_data(
    data_obj,
    model_type='RC', # String: 'RC', 'CPE', 'RC2', 'CPE2', 'RC3', 'CPE3'
                     # 'RC':   R_s + p(R1, C1+Cpad)
                     # 'CPE':  R_s + p(R1, CPE1+Cpad)
                     # 'RC2':  R_s + p(R1, C1+Cpad) + p(R2, C2)
                     # 'CPE2': R_s + p(R1, CPE1+Cpad) + p(R2, CPE2) <-- Updated
                     # 'RC3':  R_s + p( C_pad, (p(R1,C1) + p(R2,C2)) )
                     # 'CPE3': R_s + p( C_pad, (p(R1,CPE1) + p(R2,CPE2)) )
    # --- Other arguments remain the same ---
    freq_bounds=None, 
    med_filt=0, 
    fixed_params=None, 
    plot_fit=True,
    plot_type='Zrealimag', 
    fig_size=(3.5, 2.625), 
    use_de=True,
    de_bounds_dict=None, 
    ls_bounds_dict=None, 
    initial_guess_dict=None,
    de_maxiter=5000, 
    de_popsize=100, 
    de_tol=1e-5,
    ls_max_nfev=3000, 
    ls_ftol=1e-12, 
    ls_xtol=1e-12, 
    ls_gtol=1e-12
    ):
    """
    Fits impedance data to a specified equivalent circuit model...
    [Docstring Args - Update examples for CPE2 bounds/guesses]
    ...
    """
    print(f"\n--- Starting Fit for: {getattr(data_obj, 'plot_string', 'Unknown Data')} ---")
    print(f"Using model: {model_type}")
    # ... (Setup: fixed_params, print info - same as before) ...
    if fixed_params is None: fixed_params = {}
    else: print(f"With fixed params: {fixed_params}")
    if freq_bounds: print(f"Frequency range: {freq_bounds}")

    fig, ax = None, None # Initialize plot objects

    # --- Define Parameter Orders ---
    PARAM_ORDERS = { # R_s is always last
        'RC':   ['R_mem1', 'C_mem1', 'C_pad', 'R_series'],
        'CPE':  ['R_mem1', 'Q1', 'alpha1', 'C_pad', 'R_series'],
        'RC2':  ['R_mem1', 'C_mem1', 'R_mem2', 'C_mem2', 'C_pad', 'R_series'],
        'CPE2': ['R_mem1', 'Q1', 'alpha1', 'R_mem2', 'Q2', 'alpha2', 'C_pad', 'R_series'], # Added C_pad
        'RC3':  ['R_mem1', 'C_mem1', 'R_mem2', 'C_mem2', 'C_pad', 'R_series'], # Same params as RC2
        'CPE3': ['R_mem1', 'Q1', 'alpha1', 'R_mem2', 'Q2', 'alpha2', 'C_pad', 'R_series'] # Same params as CPE2
    }
    # Define Structure for Z_parameters output (Superset including C_pad, etc.)
    ALL_POSSIBLE_PARAMS = sorted(list(set(p for order in PARAM_ORDERS.values() for p in order)))

    if model_type not in PARAM_ORDERS: # ... (Error handling) ...
        print(f"Error: Invalid model_type '{model_type}'. Supported: {list(PARAM_ORDERS.keys())}")
        return fig, ax, None, False
    param_order = PARAM_ORDERS[model_type]

    # --- Validate fixed_params keys (same) ---
    valid_param_names = set(param_order);
    for key in fixed_params:
        if key not in valid_param_names: print(f"Error: Invalid fixed name '{key}'. Valid: {param_order}"); return fig, ax, None, False

    # --- Create maps/lists for free/fixed parameters (same) ---
    fixed_params_map = {param_order.index(k): v for k, v in fixed_params.items()}
    free_param_indices = [i for i, name in enumerate(param_order) if name not in fixed_params]
    free_param_names = [param_order[i] for i in free_param_indices]
    print(f"Free parameters to fit: {free_param_names}")
    if not free_param_names: # ... (Handle all fixed - same) ...
        print("All parameters are fixed..."); return fig, ax, fixed_params, True # Simplified return

    # 1. Prepare Data (same)
    plot_kernel_size = 0
    try: # ... (same data prep logic) ...
        if data_obj.Zrealimag is None: raise ValueError("Zrealimag data missing.")
        frequency_raw=data_obj.Zrealimag[:,0].copy(); Z_measured_raw=data_obj.Zrealimag[:,1].copy()+1j*data_obj.Zrealimag[:,2].copy()
        frequency=frequency_raw.copy(); Z_measured=Z_measured_raw.copy()
        if freq_bounds: min_f,max_f=freq_bounds; mask=np.ones_like(frequency,dtype=bool); mask&=(frequency>=min_f) if min_f else True; mask&=(frequency<=max_f) if max_f else True; frequency=frequency[mask]; Z_measured=Z_measured[mask]; print(f"{len(frequency)} pts post-freq filter.")
        if len(frequency)==0: raise ValueError("No data post-freq filter.")
        ks=med_filt;
        if ks and ks>1:
            if ks%2==0: ks+=1
            if len(frequency)>=ks: print(f"Median filter k={ks}"); Z_measured=medfilt(Z_measured.real,ks)+1j*medfilt(Z_measured.imag,ks); plot_kernel_size=ks
            else: print("Warn: Too few pts for median filter.")
        valid_idx=frequency>0;
        if not np.all(valid_idx): frequency=frequency[valid_idx]; Z_measured=Z_measured[valid_idx]
        if len(frequency)==0: raise ValueError("No valid freq points > 0.")
    except Exception as e: print(f"Error data prep: {e}"); return fig, ax, None, False

    # 2. Define Bounds (DE and LS) for FREE parameters
    # Update defaults to include C_pad for CPE model consistently
    default_de_bounds = {
        'R_mem1': (1e3, 1e11), 'C_mem1': (1e-13, 1e-6), 'C_pad': (1e-14, 1e-7), 'R_series': (0, 1e4),
        'Q1': (1e-14, 1e-6), 'alpha1': (0.1, 1.0),
        'R_mem2': (1e2, 1e10), 'C_mem2': (1e-14, 1e-7), 'Q2': (1e-15, 1e-7), 'alpha2': (0.1, 1.0),
    }
    default_ls_bounds = {
        'R_mem1': (1e-3, np.inf), 'C_mem1': (0, np.inf), 'C_pad': (0, np.inf), 'R_series': (0, np.inf),
        'Q1': (1e-15, np.inf), 'alpha1': (1e-3, 1.0),
        'R_mem2': (1e-3, np.inf), 'C_mem2': (0, np.inf), 'Q2': (1e-15, np.inf), 'alpha2': (1e-3, 1.0),
    }
    # ... (Merge user bounds, extract free bounds - same logic) ...
    current_de_bounds_dict = default_de_bounds.copy(); current_ls_bounds_dict = default_ls_bounds.copy()
    if de_bounds_dict: current_de_bounds_dict.update(de_bounds_dict)
    if ls_bounds_dict: current_ls_bounds_dict.update(ls_bounds_dict)
    try:
        for name in free_param_names:
             if name not in current_de_bounds_dict: raise KeyError(f"DE bound missing for '{name}'")
             if name not in current_ls_bounds_dict: raise KeyError(f"LS bound missing for '{name}'")
        de_bounds_free = [current_de_bounds_dict[name] for name in free_param_names]
        ls_bounds_free = ([current_ls_bounds_dict[name][0] for name in free_param_names],
                          [current_ls_bounds_dict[name][1] for name in free_param_names])
    except KeyError as e: print(f"Error setting bounds: {e}"); return fig, ax, None, False

    # 3 & 4: Determine Initial Guess & Run Optimizers (DE + LS)
    # ... (DE logic, Fallback guess logic - same as before) ...
    ls_initial_guess = None
    if use_de:
        print("Running DE...");
        try: # DE execution
            with warnings.catch_warnings(): warnings.simplefilter("ignore", category=RuntimeWarning)
            de_result = differential_evolution(cost_function_de, bounds=de_bounds_free, args=(frequency, Z_measured, model_type, param_order, fixed_params_map), maxiter=de_maxiter, popsize=de_popsize, tol=de_tol, disp=False)
            if de_result.success: 
                print("DE OK."); 
                ls_initial_guess = de_result.x; 
                print(f" DE Final Cost: {de_result.fun:.4e}")
                #print(f"  DE Best: {dict(zip(free_param_names, ls_initial_guess))}\n")
                print(f"  DE Best:)");  [print(f"name {free_param_names[i]}: {ls_initial_guess[i]:.2e}") for i, name in enumerate(free_param_names)]
                
            else: print(f"DE failed: {de_result.message}.")
        except Exception as e: print(f"Error DE: {e}.")

    if ls_initial_guess is None: # Fallback or if use_de=False
        if initial_guess_dict: # Try provided dict first
             try: ls_initial_guess = [initial_guess_dict[name] for name in free_param_names]; print(f"Using provided LS guess: {dict(zip(free_param_names, ls_initial_guess))}")
             except KeyError as e: print(f"Warn: Guess missing for '{e}'. Fallback."); ls_initial_guess = None
        
        if ls_initial_guess is None: # Auto-guess based on data characteristics
            print("Auto-generating intelligent initial guess for LS...")
            
            # Extract key impedance characteristics from data
            try:
                # Estimate R_series from high frequency region
                r_s_guess = np.median(Z_measured.real[frequency > 0.5 * max(frequency)]) if len(frequency)>1 else 10
                
                # Estimate R_mem from low frequency region
                r_mem_guess = np.median(Z_measured.real[frequency < 1.5 * min(frequency)])-r_s_guess if len(frequency)>1 else 1e6
                
                # Find frequency at peak imaginary component
                imag_part = -Z_measured.imag
                peak_idx = np.argmax(imag_part) if len(imag_part)>0 else 0
                f_peak = frequency[peak_idx] if len(frequency)>0 else 1e3
                omega_peak = 2*np.pi*f_peak
                
                # Create dictionary of parameter guesses
                all_guesses = {}
                all_guesses['R_mem1'] = max(r_mem_guess, 1e-2)  # First membrane resistance
                all_guesses['R_series'] = max(r_s_guess, 0)     # Series resistance
                
                # For CPE-based models
                if 'CPE' in model_type:
                    all_guesses['alpha1'] = 0.9
                    all_guesses['Q1'] = 1/(omega_peak**all_guesses['alpha1'] * all_guesses['R_mem1']) if omega_peak >0 and all_guesses['R_mem1'] > 0 else 1e-10
                    # For CPE2/CPE3 models with second element
                    if model_type in ['CPE2', 'CPE3']:
                        all_guesses['R_mem2'] = all_guesses['R_mem1'] * 0.1  # Second element usually smaller
                        all_guesses['alpha2'] = 0.8  # Often more dispersive
                        all_guesses['Q2'] = all_guesses['Q1'] * 0.5  # Different time constant
                else:
                    # For RC-based models
                    c_par_guess = 1/(omega_peak * all_guesses['R_mem1']) if omega_peak>0 and all_guesses['R_mem1']>0 else 1e-11
                    all_guesses['C_mem1'] = max(c_par_guess * 0.9, 1e-15)
                    all_guesses['C_pad'] = max(c_par_guess * 0.1, 1e-15)
                    # For RC2/RC3 models with second element
                    if model_type in ['RC2', 'RC3']:
                        all_guesses['R_mem2'] = all_guesses['R_mem1'] * 0.1  # Second element usually smaller
                        all_guesses['C_mem2'] = all_guesses['C_mem1'] * 0.5  # Different time constant
                
                # Fill in the free parameters with guesses or midpoint fallback
                low_b, high_b = ls_bounds_free
                ls_initial_guess = []
                for i, name in enumerate(free_param_names):
                    if name in all_guesses:
                        # Use intelligent guess but ensure within bounds
                        guess_val = min(max(all_guesses[name], low_b[i]), high_b[i])
                        ls_initial_guess.append(guess_val)
                    else:
                        # Fallback to midpoint for any params not in all_guesses
                        midpoint = (low_b[i]+high_b[i])/2 if np.isfinite(high_b[i]) else (low_b[i]*10 if low_b[i]>0 else 1)
                        ls_initial_guess.append(midpoint)
                
                print(f"Using auto-generated initial guess for LS: {dict(zip(free_param_names, ls_initial_guess))}")
            except Exception as e_guess:
                print(f"Error in auto-guess: {e_guess}. Falling back to midpoint...")
                try: 
                    low_b, high_b = ls_bounds_free
                    ls_initial_guess = [(l+h)/2 if np.isfinite(h) else (l*10 if l>0 else 1) for l,h in zip(low_b, high_b)]
                    print(f"Fallback Guess: {dict(zip(free_param_names, ls_initial_guess))}")
                except Exception as e_fallback:
                    print(f"Error fallback guess: {e_fallback}")
                    return fig, ax, None, False

    # --- Stage 2: Refinement with Least Squares ---
    # ... (LS call, result checking, param reconstruction - same as before) ...
    print("\nRunning LS Refinement...")
    final_fitted_params_dict = None; ls_success = False
    if ls_initial_guess is None: print("Error: No initial guess."); return fig, ax, None, False

    try:
        ls_initial_guess = np.maximum(ls_initial_guess, ls_bounds_free[0]); ls_initial_guess = np.minimum(ls_initial_guess, ls_bounds_free[1])
        ls_result = least_squares(residuals_ls_weighted, ls_initial_guess, args=(frequency, Z_measured, model_type, param_order, fixed_params_map), bounds=ls_bounds_free, method='trf', ftol=ls_ftol, xtol=ls_xtol, gtol=ls_gtol, max_nfev=ls_max_nfev)

        if ls_result.success:
            ls_success = True; print("LS OK!")
            fitted_params_free = ls_result.x; final_cost = ls_result.cost; print(f"  LS Final Cost: {final_cost:.4e}")
            final_fitted_params_dict = fixed_params.copy(); final_fitted_params_dict.update(dict(zip(free_param_names, fitted_params_free)))
            print("  Final Fitted Parameters:"); [print(f"    {name}: {final_fitted_params_dict[name]:.4e}") for name in param_order]

            # --- Store parameters in data_obj.Z_parameters ---
            target_Z_params = {name: None for name in ALL_POSSIBLE_PARAMS}
            for name, value in final_fitted_params_dict.items():
                 if name in target_Z_params: target_Z_params[name] = value
            data_obj.Z_parameters = target_Z_params
            #print(f"Stored fitted parameters in data_obj.Z_parameters")

            # --- Calculate final fit curve over FULL frequency range ---
            Z_complex_fit_full = None
            try:
                fitted_params_full_vector = [final_fitted_params_dict[name] for name in param_order]
                if frequency_raw is not None and len(frequency_raw) > 0:
                     Z_complex_fit_full = unified_circuit_model(fitted_params_full_vector, frequency_raw, model_type)
                     data_obj.Zcomplex_fit = np.column_stack((frequency_raw, Z_complex_fit_full))
                     #print("Stored extrapolated fit curve in data_obj.Zcomplex_fit")
                else: print("Warn: Cannot calculate full fit curve.")
            except Exception as e_fc: print(f"Warn: Error calculating final fit curve: {e_fc}")

              # --- Plotting ---
            if plot_fit:
                if Z_complex_fit_full is not None: # Only plot if fit curve was calculated
                    fit_label = f"{model_type} Fit (LS)"
                    if use_de: fit_label = f"{model_type} Fit (DE+LS)"

                    double_fig_size = (fig_size[0]*2, fig_size[1])  # Adjusted for two subplots
                    fig, ax = plt.subplots(1, 2, figsize=double_fig_size, sharex=True, constrained_layout=True)
                    scale_factor = 1 # No scaling initially, adjust if needed based on data
                    y_unit_prefix = ""
                    # Auto-detect scale based on max Z'
                    max_z_real = np.max(np.abs(Z_measured_raw.real)) if len(Z_measured_raw)>0 else 1
                    if max_z_real > 2e6: scale_factor = 1e6; y_unit = r"M$\Omega$"
                    elif max_z_real > 2e3: scale_factor = 1e3; y_unit = r"k$\Omega$"
                    else: scale_factor = 1; y_unit = r"$\Omega$"

                    if plot_type == 'Zrealimag':
                        # measured data over full range
                        y1_measured_raw = Z_measured_raw.real / scale_factor
                        y2_measured_raw = -Z_measured_raw.imag / scale_factor
                        x_measured_raw = frequency_raw
                        
                        # measured data over fitted range
                        y1_measured = Z_measured.real / scale_factor
                        y2_measured = -Z_measured.imag / scale_factor
                        x_measured = frequency
                        
                        # fitted data
                        y1_fit = Z_complex_fit_full.real / scale_factor
                        y2_fit = -Z_complex_fit_full.imag / scale_factor
                        x_fit = frequency_raw
                        
                        ax[0].set_ylabel(f"$Z'$ ({y_unit})")
                        ax[1].set_ylabel(f"$-Z''$ ({y_unit})")
                        
                    elif plot_type == 'Zabsphi':
                        # measured data over full range
                        y1_measured_raw = np.abs(Z_measured_raw) / scale_factor
                        y2_measured_raw = np.angle(Z_measured_raw, deg=True)
                        x_measured_raw = frequency_raw
                        
                        # measured data over fitted range
                        y1_measured = np.abs(Z_measured) / scale_factor
                        y2_measured = np.angle(Z_measured, deg=True)
                        x_measured = frequency
                        
                        # fitted data
                        y1_fit = np.abs(Z_complex_fit_full) / scale_factor
                        y2_fit = np.angle(Z_complex_fit_full, deg=True)
                        x_fit = frequency_raw
                        
                        ax[0].set_ylabel(f"$|Z|$ ({y_unit})")
                        ax[1].set_ylabel(r"Phase ($^{\circ}$)")
                    
                    # measured full range
                    ax[0].plot(x_measured_raw, y1_measured_raw, 'o', ms=3, color='lightgrey', label='_nolegend_')
                    ax[1].plot(x_measured_raw, y2_measured_raw, 'o', ms=3, color='lightgrey', label='_nolegend_')
                    
                    #measured fitted range
                    ax[0].plot(x_measured, y1_measured, 'o', ms=5, label='Measured (used)')
                    ax[1].plot(x_measured, y2_measured, 'o', ms=5, label='Measured (used)')
                    
                    # Fitted 
                    ax[0].plot(x_fit, y1_fit, '-', lw=2, label=fit_label)
                    ax[1].plot(x_fit, y2_fit, '-', lw=2, label=fit_label)

                    ax[0].set_xlabel("Frequency (Hz)"); ax[0].legend()
                    ax[0].set_xlim(min(frequency_raw)*0.8, max(frequency_raw)*1.2); ax[0].set_xscale('log'); ax[0].set_yscale('log') # Log scale Y for Z'
                    ax[1].set_xlabel("Frequency (Hz)"); ax[1].legend()
                    # ax[1].set_yscale('log') # Often better linear for -Z'' peak

                    title_note = ""
                    if freq_bounds is not None: title_note += f" Freq: {freq_bounds}"
                    if plot_kernel_size > 1: title_note += f" Filter: k={plot_kernel_size}"
                    if fixed_params: title_note += f" Fixed: {fixed_params}"
                    plot_string_val = getattr(data_obj, 'plot_string', 'Unknown Data')
                    if plot_string_val is None: plot_string_val = 'Unknown Data'
                    fig.suptitle(f'{fit_label} for {plot_string_val}{title_note}')
                    plt.show()
                else:
                     print("Skipping plot: Failed to calculate final fit curve.")

                print("\nFinal Fitted Parameters Dictionary:")
                print(final_fitted_params_dict)



        else: # LS failed
            print(f"LS failed: {ls_result.message}")
            ls_success = False

    except ValueError as ve: print(f"ValueError LS stage: {ve}"); ls_success = False
    except Exception as e: print(f"Unexpected error LS stage: {e}"); ls_success = False

    print(f"--- Fit finished for: {getattr(data_obj, 'plot_string', 'Unknown Data')} ---")
    return fig, ax