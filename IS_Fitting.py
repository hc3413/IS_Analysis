#import all the libraries needed
from import_dep import *


# --- Unified Model Function (Handles Full or Free Params, RC, CPE, RC2, CPE2) ---
def unified_circuit_model(params, frequency, model_type='RC', param_order=None, fixed_params_map=None):
    """
    Calculates impedance for various equivalent circuit models.
    Handles fixed parameters by reconstructing the full parameter list if needed.

    Supported models:
    'RC':   R_s + p(R_mem1, C_mem1 + C_pad)
    'CPE':  R_s + p(R_mem1, CPE1) where CPE1 = (Q1, alpha1)
    'RC2':  R_s + p(R_mem1, C_mem1 + C_pad) + p(R_mem2, C_mem2)
    'CPE2': R_s + p(R_mem1, CPE1) + p(R_mem2, CPE2)

    Args:
        params (list/tuple): EITHER the FULL parameter list OR only the FREE parameters.
                             Order depends on model_type. See PARAM_ORDERS below.
        frequency (np.ndarray): Array of frequencies (Hz).
        model_type (str): 'RC', 'CPE', 'RC2', or 'CPE2'.
        param_order (list | None): Full order of parameter names. Needed if params contains only free params.
        fixed_params_map (dict | None): Map of fixed param index to value. Needed if params contains only free params.

    Returns:
        np.ndarray: Complex impedance Z_model.
    """
    # Define parameter orders locally for clarity
    PARAM_ORDERS = {
        'RC':   ['R_mem1', 'C_mem1', 'C_pad', 'R_series'],
        'CPE':  ['R_mem1', 'Q1', 'alpha1', 'R_series'],
        'RC2':  ['R_mem1', 'C_mem1', 'R_mem2', 'C_mem2', 'C_pad', 'R_series'], # C_pad associated with first RC
        'CPE2': ['R_mem1', 'Q1', 'alpha1', 'R_mem2', 'Q2', 'alpha2', 'R_series']
    }
    if model_type not in PARAM_ORDERS:
        raise ValueError(f"Unknown model_type '{model_type}'. Supported types: {list(PARAM_ORDERS.keys())}")

    current_param_order = PARAM_ORDERS[model_type]
    expected_len = len(current_param_order)
    full_params = None

    # --- Check if reconstruction is needed ---
    if param_order is not None and fixed_params_map is not None:
        # Ensure the passed param_order matches the expected one for the model_type
        if param_order != current_param_order:
             raise ValueError(f"param_order provided does not match expected order for model_type '{model_type}'")
        # Assume params contains ONLY FREE parameters - reconstruct full list
        if len(param_order) != len(params) + len(fixed_params_map):
             raise ValueError(f"Parameter length mismatch during reconstruction. "
                              f"Order={len(param_order)}, Free={len(params)}, Fixed={len(fixed_params_map)}")
        full_params = [None] * len(param_order)
        free_param_idx = 0
        for i in range(len(param_order)):
            if i in fixed_params_map:
                full_params[i] = fixed_params_map[i]
            else:
                if free_param_idx >= len(params): raise IndexError("More free params expected.")
                full_params[i] = params[free_param_idx]
                free_param_idx += 1
        if free_param_idx != len(params): raise ValueError("Did not use all free params.")
    else:
        # Assume params already contains the FULL parameter list
        full_params = list(params)
    # --- End Reconstruction Logic ---

    # --- Validate length of full_params ---
    if len(full_params) != expected_len:
        raise ValueError(f"Incorrect number of parameters in full_params for model '{model_type}'. "
                         f"Expected {expected_len}, got {len(full_params)}.")

    # --- Calculations ---
    omega = 2 * np.pi * frequency
    epsilon = 1e-18
    Z_parallel1 = 0
    Z_parallel2 = 0
    R_series_val = 0

    # Define helper for parallel impedance Zp = 1 / (1/R + Y_element)
    def calculate_parallel_Z(R_val, Y_elem_val, eps):
        R_safe = max(R_val, eps)
        with np.errstate(divide='ignore', invalid='ignore'):
            Y_R = 1 / R_safe
            Z_inv = Y_R + Y_elem_val
            Z_inv_safe = np.where(np.abs(Z_inv) < eps, eps, Z_inv)
            Z_p = 1 / Z_inv_safe
            return np.nan_to_num(Z_p, nan=np.inf)

    if model_type == 'RC':
        R_mem1, C_mem1, C_pad, R_series = full_params
        C_par = max(C_mem1, 0) + max(C_pad, 0)
        Y_elem = 1j * omega * C_par
        Z_parallel1 = calculate_parallel_Z(R_mem1, Y_elem, epsilon)
        R_series_val = max(R_series, 0)

    elif model_type == 'CPE':
        R_mem1, Q1, alpha1, R_series = full_params
        Q1_safe = max(Q1, epsilon); alpha1_safe = max(min(alpha1, 1.0), epsilon)
        Y_elem = Q1_safe * (1j * omega)**alpha1_safe
        Z_parallel1 = calculate_parallel_Z(R_mem1, Y_elem, epsilon)
        R_series_val = max(R_series, 0)

    elif model_type == 'RC2':
        R_mem1, C_mem1, R_mem2, C_mem2, C_pad, R_series = full_params
        # First parallel element (associating C_pad here)
        C_par1 = max(C_mem1, 0) + max(C_pad, 0)
        Y_elem1 = 1j * omega * C_par1
        Z_parallel1 = calculate_parallel_Z(R_mem1, Y_elem1, epsilon)
        # Second parallel element
        C_par2 = max(C_mem2, 0)
        Y_elem2 = 1j * omega * C_par2
        Z_parallel2 = calculate_parallel_Z(R_mem2, Y_elem2, epsilon)
        R_series_val = max(R_series, 0)

    elif model_type == 'CPE2':
        R_mem1, Q1, alpha1, R_mem2, Q2, alpha2, R_series = full_params
        # First parallel element
        Q1_safe = max(Q1, epsilon); alpha1_safe = max(min(alpha1, 1.0), epsilon)
        Y_elem1 = Q1_safe * (1j * omega)**alpha1_safe
        Z_parallel1 = calculate_parallel_Z(R_mem1, Y_elem1, epsilon)
        # Second parallel element
        Q2_safe = max(Q2, epsilon); alpha2_safe = max(min(alpha2, 1.0), epsilon)
        Y_elem2 = Q2_safe * (1j * omega)**alpha2_safe
        Z_parallel2 = calculate_parallel_Z(R_mem2, Y_elem2, epsilon)
        R_series_val = max(R_series, 0)

    # Combine elements
    Z_model = R_series_val + Z_parallel1 + Z_parallel2 # Z_parallel2 is 0 for RC/CPE

    return Z_model

# --- Weighted Residual Function (No change needed structurally) ---
def residuals_ls_weighted(params, frequency, Z_measured, model_type, param_order, fixed_params_map):
    """ Residual function wrapper for least_squares handling fixed params """
    Z_model = unified_circuit_model(params, frequency, model_type, param_order, fixed_params_map)
    Z_diff = Z_measured - Z_model
    epsilon_weight = 1e-9
    mod_Z_measured = np.abs(Z_measured)
    weights = 1.0 / np.maximum(mod_Z_measured, epsilon_weight * np.mean(mod_Z_measured) + epsilon_weight)
    residuals_real = Z_diff.real * weights
    residuals_imag = Z_diff.imag * weights
    return np.concatenate((residuals_real, residuals_imag))

# --- Cost Function (No change needed structurally) ---
def cost_function_de(params, frequency, Z_measured, model_type, param_order, fixed_params_map):
    """ Cost function wrapper for DE handling fixed params """
    residuals = residuals_ls_weighted(params, frequency, Z_measured, model_type, param_order, fixed_params_map)
    return np.sum(residuals**2)


# --- Main Fitting Function ---
def fit_impedance_data(
    data_obj, # The ISdata object containing impedance data (must have .Zrealimag)
    model_type='RC', # String: 'RC', 'CPE', 'RC2', or 'CPE2'
                     # 'RC':   R_s + p(R_mem1, C_mem1 + C_pad)
                     # 'CPE':  R_s + p(R_mem1, CPE1)
                     # 'RC2':  R_s + p(R_mem1, C_mem1 + C_pad) + p(R_mem2, C_mem2)
                     # 'CPE2': R_s + p(R_mem1, CPE1) + p(R_mem2, CPE2)
    freq_bounds=None, # Tuple (min_freq, max_freq) for fitting range. None means no limit.
                      # freq_bounds = (1e1, 1e5)
    med_filt=0, # Odd integer kernel size > 1 for median filter. 0 or None disables.
                # med_filt = 3
    fixed_params=None, # Dictionary mapping parameter names (str) to fixed values.
                       # fixed_params = {'R_series': 55.0}
                       # fixed_params = {'C_pad': 1.2e-11, 'alpha1': 0.9}
    plot_fit=True, # If True, display the Bode plot of the fit after completion.
    plot_type = 'Zabsphi', # Plot type: 'Zrealimag' or 'Zabsphi'
    fig_size: tuple = (3.5, 2.625), # Base size of the figure
    use_de=True, # If True, perform Differential Evolution before Least Squares.
    de_bounds_dict=None, # Dictionary mapping param names to (min, max) tuples for DE search space.
                         # If None, uses broad internal defaults. More specific bounds recommended.
                         # de_bounds_dict:{'R_mem1': (1e5, 1e8), 'C_mem1': (1e-11, 1e-8),'R_mem2': (1e3, 1e7), 'C_mem2': (1e-10, 1e-7),
                                        #  'C_pad': (1e-12, 1e-9), 'R_series': (10, 500), 'Q1': (1e-12, 1e-7), 'alpha1': (0.6, 1.0),
                         #               'Q2': (1e-13, 1e-8), 'alpha2': (0.5, 1.0)}
                        
    ls_bounds_dict=None, # Dictionary mapping param names to (min, max) tuples for LS refinement.
                         # If None, uses broad internal defaults (usually (0/epsilon, inf)).
                         # ls_bounds_dict = {'R_mem1': (1e-3, np.inf), 'C_mem1': (0, np.inf), 'R_mem2': (1e-3, np.inf), 
                                            # 'C_mem2': (0, np.inf), 'C_pad': (0, np.inf), 'R_series': (0, np.inf), 'Q1': (1e-15, np.inf), 
                                            # 'alpha1': (1e-3, 1.0), 'Q2': (1e-15, np.inf), 'alpha2': (1e-3, 1.0)}
    initial_guess_dict=None, # Dictionary mapping param names to initial guess values for LS.
                             # Only used if use_de=False OR if DE fails. If None, uses auto-guess.
                             # initial_guess_dict = {'R_mem1': 5e7, 'C_mem1': 5e-11, 'R_mem2': 1e5, 'C_mem2': 1e-9, 'C_pad': 2e-11, 'R_series': 70.0,
                                                    # 'Q1': 4e-11, 'alpha1': 0.92, 'Q2': 5e-10, 'alpha2': 0.85}
    de_maxiter=1500, # Max generations (iterations) for Differential Evolution.
    ls_max_nfev=10000, # Max function evaluations for Least Squares.
    de_popsize=80, # Population size for Differential Evolution.
    de_tol=1e-4, # Tolerance for Differential Evolution.
    ls_ftol=1e-12, # Tolerance for Least Squares (ftol).
    ls_xtol=1e-12, # Tolerance for Least Squares (xtol).
    ls_gtol=1e-12, # Tolerance for Least Squares (gtol).
    ):
    """
    Fits impedance data to a specified equivalent circuit model using
    optional filtering, fixed parameters, and optimization stages.
    Stores results in data_obj.Z_parameters and data_obj.Zcomplex_fit.

    [Rest of Docstring Args...]

    Returns:
        tuple: (fitted_params_dict, fit_success)
               - fitted_params_dict (dict | None): Final parameters (fixed & fitted), or None on failure.
               - fit_success (bool): True if the Least Squares refinement step was successful.
    """
    print(f"\n--- Starting Fit for: {getattr(data_obj, 'plot_string', 'Unknown Data')} ---")
    print(f"Using model: {model_type}")
    if fixed_params is None:
        fixed_params = {}
    else:
         print(f"With fixed params: {fixed_params}")
    if freq_bounds: print(f"Frequency range: {freq_bounds}")

    # --- Define Parameter Orders ---
    PARAM_ORDERS = {
        'RC':   ['R_mem1', 'C_mem1', 'C_pad', 'R_series'],
        'CPE':  ['R_mem1', 'Q1', 'alpha1', 'R_series'],
        'RC2':  ['R_mem1', 'C_mem1', 'R_mem2', 'C_mem2', 'C_pad', 'R_series'],
        'CPE2': ['R_mem1', 'Q1', 'alpha1', 'R_mem2', 'Q2', 'alpha2', 'R_series']
    }
    if model_type not in PARAM_ORDERS:
        print(f"Error: Invalid model_type '{model_type}'. Supported: {list(PARAM_ORDERS.keys())}")
        return None, False
    param_order = PARAM_ORDERS[model_type]

    # --- Validate fixed_params keys ---
    valid_param_names = set(param_order)
    for key in fixed_params:
        if key not in valid_param_names:
            print(f"Error: Invalid parameter name '{key}' in fixed_params for model '{model_type}'. "
                  f"Valid names are: {param_order}")
            return None, False

    # --- Create maps/lists for free/fixed parameters ---
    fixed_params_map = {param_order.index(k): v for k, v in fixed_params.items()}
    free_param_indices = [i for i, name in enumerate(param_order) if name not in fixed_params]
    free_param_names = [param_order[i] for i in free_param_indices]
    print(f"Free parameters to fit: {free_param_names}")
    if not free_param_names:
        print("Warning: All parameters are fixed. Cannot perform fit.")
        # Calculate model with fixed params and store
        try:
             frequency_raw = data_obj.Zrealimag[:, 0].copy()
             fixed_vals_vector = [fixed_params[name] for name in param_order]
             Z_complex_fit_full = unified_circuit_model(fixed_vals_vector, frequency_raw, model_type)
             # Store as (freq, Z) where Z is complex
             data_obj.Zcomplex_fit = np.column_stack((frequency_raw, Z_complex_fit_full))
             data_obj.Z_parameters = {name:fixed_params.get(name) for name in PARAM_ORDERS['CPE2']} # Store in full structure
             print("Stored model curve based on fixed parameters.")
             return fixed_params, True # Return fixed params, indicate success as calculation done
        except Exception as e:
             print(f"Error calculating model with fixed parameters: {e}")
             return None, False


    # 1. Prepare Data (Load, Filter Freq, Apply Median Filter)
    plot_kernel_size = 0 # Initialize for plotting title
    try:
        if data_obj.Zrealimag is None: raise ValueError("Zrealimag data is missing.")
        frequency_raw = data_obj.Zrealimag[:, 0].copy()
        Z_measured_raw = data_obj.Zrealimag[:, 1].copy() + 1j * data_obj.Zrealimag[:, 2].copy()

        frequency = frequency_raw.copy(); Z_measured = Z_measured_raw.copy()
        freq_mask = np.ones_like(frequency, dtype=bool)
        # ... (Frequency filtering logic - same as before) ...
        if freq_bounds is not None:
            min_freq, max_freq = freq_bounds
            if min_freq is not None: freq_mask &= (frequency >= min_freq)
            if max_freq is not None: freq_mask &= (frequency <= max_freq)
            frequency = frequency[freq_mask]; Z_measured = Z_measured[freq_mask]
            print(f"{len(frequency)} points after frequency filtering.")
        if len(frequency) == 0: raise ValueError("No data after frequency filtering.")

        kernel_size = med_filt
        if kernel_size is not None and kernel_size > 1:
             if kernel_size % 2 == 0: kernel_size += 1
             if len(frequency) >= kernel_size:
                 print(f"Applying median filter (k={kernel_size})...")
                 Z_measured = medfilt(Z_measured.real, kernel_size=kernel_size) + 1j * medfilt(Z_measured.imag, kernel_size=kernel_size)
                 plot_kernel_size = kernel_size # Store for plot title
             else: print("Warning: Insufficient points for median filter. Skipping.")

        valid_freq_indices = frequency > 0
        if not np.all(valid_freq_indices):
             frequency = frequency[valid_freq_indices]; Z_measured = Z_measured[valid_freq_indices]
        if len(frequency) == 0: raise ValueError("No valid frequency points > 0.")

    except Exception as e:
        print(f"Error during data preparation: {e}")
        return None, False

    # 2. Define Bounds (DE and LS) for FREE parameters
    # Consolidate default bounds - add new params here
    default_de_bounds = {
        'C_pad': (1e-13, 1e-6),'R_series': (0, 1e4),
        'R_mem1': (1e3, 1e11), 'C_mem1': (1e-13, 1e-6),
        'R_mem2': (1e2, 1e10), 'C_mem2': (1e-14, 1e-7), # Example ranges
        'Q1': (1e-14, 1e-6), 'alpha1': (0.5, 1.0),
        'Q2': (1e-15, 1e-7), 'alpha2': (0.5, 1.0)
    }
    default_ls_bounds = {
        'C_pad': (0, np.inf), 'R_series': (0, np.inf),
        'R_mem1': (1e-3, np.inf), 'C_mem1': (0, np.inf),
        'R_mem2': (1e-3, np.inf), 'C_mem2': (0, np.inf),
        'Q1': (1e-15, np.inf), 'alpha1': (1e-3, 1.0),
        'Q2': (1e-15, np.inf), 'alpha2': (1e-3, 1.0)
    }

    current_de_bounds_dict = de_bounds_dict if de_bounds_dict is not None else default_de_bounds
    current_ls_bounds_dict = ls_bounds_dict if ls_bounds_dict is not None else default_ls_bounds

    try:
        # Check if all needed free params are in bounds dicts
        for name in free_param_names:
             if name not in current_de_bounds_dict: raise KeyError(f"DE bound missing for '{name}'")
             if name not in current_ls_bounds_dict: raise KeyError(f"LS bound missing for '{name}'")

        de_bounds_free = [current_de_bounds_dict[name] for name in free_param_names]
        ls_bounds_free = ([current_ls_bounds_dict[name][0] for name in free_param_names],
                          [current_ls_bounds_dict[name][1] for name in free_param_names])
    except KeyError as e:
        print(f"Error setting bounds: {e}. Make sure de/ls_bounds_dict includes all free parameters.")
        return None, False

    # --- Determine Initial Guess for FREE parameters ---
    # (Keeping DE->LS flow, adding basic auto-guess for new models)
    ls_initial_guess = None
    if use_de:
        print("Running Differential Evolution (Stage 1)...")
        try:
            with warnings.catch_warnings():
                 warnings.simplefilter("ignore", category=RuntimeWarning)
                 de_result = differential_evolution(
                    cost_function_de, bounds=de_bounds_free,
                    args=(frequency, Z_measured, model_type, param_order, fixed_params_map),
                    strategy='best1bin', maxiter=de_maxiter, popsize=de_popsize, tol=de_tol,
                    mutation=(0.5, 1), recombination=0.7, updating='immediate', disp=False)
            if de_result.success:
                print("DE finished successfully.")
                ls_initial_guess = de_result.x
                print(f"  DE Best Params (Free): {dict(zip(free_param_names, ls_initial_guess))}")
                print(f"  DE Final Cost: {de_result.fun:.4e}")
            else: print(f"DE failed: {de_result.message}. Attempting LS with fallback guess.")
        except Exception as e: print(f"Error during DE: {e}. Attempting LS with fallback guess.")

    if ls_initial_guess is None: # Fallback or if use_de=False
         # ... (Auto-guess logic - needs update for RC2/CPE2) ...
         # For now, we'll just use mid-point of LS bounds as a basic fallback
         print("Using fallback initial guess (midpoint of LS bounds) for LS...")
         try:
             low_b = ls_bounds_free[0]
             high_b = ls_bounds_free[1]
             # Calculate midpoint, handling potential infinities
             ls_initial_guess = []
             for l, h in zip(low_b, high_b):
                 if np.isinf(h):
                     mid = l * 10 if l > 0 else 1 # Guess slightly above lower bound if upper is inf
                 elif np.isinf(l):
                      mid = h * 0.1 if h > 0 else -1 # Guess slightly below upper bound if lower is inf
                 else:
                      mid = (l+h)/2
                 ls_initial_guess.append(mid)
             print(f"Fallback Guess (Free): {dict(zip(free_param_names, ls_initial_guess))}")
         except Exception as e_guess:
              print(f"Error creating fallback guess: {e_guess}. Cannot proceed.")
              return None, False


    # --- Stage 2: Refinement with Least Squares ---
    print("\nRunning Least Squares Refinement (Stage 2)...")
    final_fitted_params_dict = None # Initialize dictionary for final results
    ls_success = False
    if ls_initial_guess is None:
         print("Error: No valid initial guess available for Least Squares.")
         return None, False # Already returned above, but double-check

    try:
        # Ensure initial guess is within LS bounds
        ls_initial_guess = np.maximum(ls_initial_guess, ls_bounds_free[0])
        ls_initial_guess = np.minimum(ls_initial_guess, ls_bounds_free[1])

        ls_result = least_squares(
            residuals_ls_weighted, ls_initial_guess,
            args=(frequency, Z_measured, model_type, param_order, fixed_params_map),
            bounds=ls_bounds_free, method='trf',
            ftol=ls_ftol, xtol=ls_xtol, gtol=ls_gtol, max_nfev=ls_max_nfev)

        if ls_result.success:
            ls_success = True
            print("Least Squares refinement successful!")
            fitted_params_free = ls_result.x
            final_cost = ls_result.cost
            print(f"  LS Final Cost (0.5*sum(sq_resid)): {final_cost:.4e}")

            # --- Reconstruct full parameter dictionary ---
            final_fitted_params_dict = fixed_params.copy() # Start with fixed values
            final_fitted_params_dict.update(dict(zip(free_param_names, fitted_params_free)))

            print("  Final Fitted Parameters:")
            for name in param_order: print(f"    {name}: {final_fitted_params_dict[name]:.4e}")

            # --- Store parameters in data_obj.Z_parameters ---
            # Initialize the target dictionary with all possible keys
            target_Z_params = {
                'C_pad':None, 'R_series':None,
                'R_mem1':None, 'C_mem1':None, 'R_mem2':None, 'C_mem2':None,
                'Q1':None, 'alpha1':None, 'Q2':None, 'alpha2':None
            }
            # Populate with values from the current fit
            for name, value in final_fitted_params_dict.items():
                 if name in target_Z_params:
                     target_Z_params[name] = value
            data_obj.Z_parameters = target_Z_params
            print(f"Stored fitted parameters in data_obj.Z_parameters")


            # --- Calculate final fit curve over FULL frequency range ---
            try:
                fitted_params_full_vector = [final_fitted_params_dict[name] for name in param_order]
                if frequency_raw is not None and len(frequency_raw) > 0:
                     Z_complex_fit_full = unified_circuit_model(fitted_params_full_vector, frequency_raw, model_type)
                     # Store as (freq, Z) where Z is complex
                     data_obj.Zcomplex_fit = np.column_stack((frequency_raw, Z_complex_fit_full))
                     print("Stored extrapolated fit curve in data_obj.Zcomplex_fit")
                else: Z_complex_fit_full = None # Indicate calc failed for plot
            except Exception as e_fitcurve:
                 print(f"Warning: Error calculating final fit curve: {e_fitcurve}")
                 Z_complex_fit_full = None # Indicate calc failed for plot


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
            print(f"Least Squares refinement failed: {ls_result.message}")
            ls_success = False


    except ValueError as ve: print(f"ValueError during LS: {ve}"); ls_success = False
    except Exception as e: print(f"Unexpected error during LS: {e}"); ls_success = False

    print(f"--- Fit finished for: {getattr(data_obj, 'plot_string', 'Unknown Data')} ---")
    return fig, ax



 

    print(f"--- Fit finished for: {data_obj.plot_string} ---")
    return fitted_params_dict, ls_success