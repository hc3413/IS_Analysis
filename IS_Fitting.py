#import all the libraries needed
from import_dep import *

# --- Unified Model Function (Handles Full or Free Params) ---
def unified_circuit_model(params, frequency, model_type='RC', param_order=None, fixed_params_map=None):
    """
    Calculates impedance for either RC or CPE parallel element model.
    Handles fixed parameters by reconstructing the full parameter list if needed.

    Args:
        params (list/tuple): EITHER the FULL parameter list OR only the FREE parameters.
                             If param_order and fixed_params_map are provided,
                             params is assumed to be the FREE parameters. Otherwise,
                             it's assumed to be the FULL parameter list.
        frequency (np.ndarray): Array of frequencies (Hz).
        model_type (str): 'RC' or 'CPE'.
        param_order (list | None): Full order of parameter names. Needed if params contains only free params.
        fixed_params_map (dict | None): Map of fixed param index to value. Needed if params contains only free params.

    Returns:
        np.ndarray: Complex impedance Z_model.
    """
    full_params = None
    # --- Check if reconstruction is needed ---
    if param_order is not None and fixed_params_map is not None:
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
        full_params = list(params) # Ensure it's a mutable list if needed
    # --- End Reconstruction Logic ---

    # --- Proceed using full_params ---
    expected_len = 4 # Both models have 4 parameters
    if len(full_params) != expected_len:
        raise ValueError(f"Incorrect number of parameters in full_params for model {model_type}. "
                         f"Expected {expected_len}, got {len(full_params)}.")

    omega = 2 * np.pi * frequency
    epsilon = 1e-18

    if model_type == 'RC':
        R_mem, C_mem, C_pad, R_series = full_params # Unpack from potentially reconstructed list
        R_mem_safe = max(R_mem, epsilon); C_mem_safe = max(C_mem, 0); C_pad_safe = max(C_pad, 0); R_series_safe = max(R_series, 0)
        C_par = C_mem_safe + C_pad_safe
        Y_element = 1j * omega * C_par
    elif model_type == 'CPE':
        R_mem, Q, alpha, R_series = full_params # Unpack from potentially reconstructed list
        R_mem_safe = max(R_mem, epsilon); Q_safe = max(Q, epsilon); alpha_safe = max(min(alpha, 1.0), epsilon); R_series_safe = max(R_series, 0)
        Y_element = Q_safe * (1j * omega)**alpha_safe
    else:
        raise ValueError("Unknown model_type specified.")

    # --- Calculation (Remains the same) ---
    with np.errstate(divide='ignore', invalid='ignore'):
        Y_R = 1 / R_mem_safe
        Z_parallel_inv = Y_R + Y_element
        Z_parallel_inv_safe = np.where(np.abs(Z_parallel_inv) < epsilon, epsilon, Z_parallel_inv)
        Z_parallel = 1 / Z_parallel_inv_safe
    Z_model = R_series_safe + Z_parallel
    Z_model = np.nan_to_num(Z_model, nan=np.inf)
    return Z_model

# --- Weighted Residual Function (Takes free params, calls model with reconstruction) ---
def residuals_ls_weighted(params, frequency, Z_measured, model_type, param_order, fixed_params_map):
    """ Residual function wrapper for least_squares handling fixed params """
    Z_model = unified_circuit_model(params, frequency, model_type, param_order, fixed_params_map)
    Z_diff = Z_measured - Z_model
    epsilon_weight = 1e-9
    mod_Z_measured = np.abs(Z_measured)
    # Robust weighting, preventing weights from becoming infinite if |Z| is tiny
    weights = 1.0 / np.maximum(mod_Z_measured, epsilon_weight * np.mean(mod_Z_measured) + epsilon_weight)
    residuals_real = Z_diff.real * weights
    residuals_imag = Z_diff.imag * weights
    return np.concatenate((residuals_real, residuals_imag))

# --- Cost Function (Takes free params, calls model with reconstruction) ---
def cost_function_de(params, frequency, Z_measured, model_type, param_order, fixed_params_map):
    """ Cost function wrapper for DE handling fixed params """
    residuals = residuals_ls_weighted(params, frequency, Z_measured, model_type, param_order, fixed_params_map)
    return np.sum(residuals**2)



# --- Main Fitting Function ---
def fit_impedance_data(
    data_obj,
    use_cpe_model=False,
    freq_bounds=None,
    med_filt=0,
    fixed_params=None, # Dict like {'C_pad': 1e-12, 'R_series': 50}
    plot_fit=True,
    use_de=True,
    de_bounds_dict=None, # Provide specific bounds for DE
    ls_bounds_dict=None, # Provide specific bounds for LS
    initial_guess_dict=None, # Optional initial guess dict
    de_maxiter=500,
    ls_max_nfev=3000
    ):
    """
    Fits impedance data to an equivalent circuit model (RC or CPE) using
    optional median filtering, frequency range limiting, fixed parameters,
    and a two-stage (DE + LS) optimization approach.

    Args:
        data_obj: ISdata object with .Zrealimag attribute.
        use_cpe_model (bool): If True, use Rs+p(R,CPE) model. If False, use Rs+p(R,C+C).
        freq_bounds (tuple | None): (min_freq, max_freq) for fitting range. None means no limit.
        med_filt (int | None): Odd integer kernel size for median filter. 0 or None disables.
        fixed_params (dict | None): Dictionary mapping parameter names (str) to fixed values.
                                    e.g., {'C_pad': 1e-12}. Parameter names must match
                                    standard order ('R_mem', 'C_mem', 'C_pad', 'R_series' or
                                    'R_mem', 'Q', 'alpha', 'R_series').
        plot_fit (bool): If True, display the Bode plot of the fit.
        use_de (bool): If True, perform Differential Evolution before Least Squares.
        de_bounds_dict (dict | None): Bounds for DE. Maps param names to (min, max).
                                     If None, uses broad default bounds.
        ls_bounds_dict (dict | None): Bounds for LS. Maps param names to (min, max).
                                     If None, uses broad default bounds.
        initial_guess_dict (dict | None): Initial guess. Maps param names to values.
                                         If None or DE is used, auto-guesses/uses DE result.
        de_maxiter (int): Max iterations for Differential Evolution.
        ls_max_nfev (int): Max function evaluations for Least Squares.


    Returns:
        tuple: (fitted_params_dict, fit_success)
               - fitted_params_dict (dict): Dictionary of final fitted parameter values.
                                            Returns None if fitting fails severely.
               - fit_success (bool): True if the Least Squares refinement step was successful.
    """
    print(f"\n--- Starting Fit for: {data_obj.plot_string} ---")
    if fixed_params is None:
        fixed_params = {}

    # --- Define Parameter Order based on Model ---
    RC_PARAM_ORDER = ['R_mem', 'C_mem', 'C_pad', 'R_series']
    CPE_PARAM_ORDER = ['R_mem', 'Q', 'alpha', 'R_series']
    if use_cpe_model:
        model_type_str = 'CPE'
        param_order = CPE_PARAM_ORDER
        print("Using CPE model.")
    else:
        model_type_str = 'RC'
        param_order = RC_PARAM_ORDER
        print("Using RC model.")

    # --- Validate fixed_params keys ---
    valid_param_names = set(param_order)
    for key in fixed_params:
        if key not in valid_param_names:
            raise ValueError(f"Invalid parameter name '{key}' in fixed_params. "
                             f"Valid names for {model_type_str} are: {param_order}")

    # --- Create maps/lists for free/fixed parameters ---
    fixed_params_map = {param_order.index(k): v for k, v in fixed_params.items()}
    free_param_indices = [i for i, name in enumerate(param_order) if name not in fixed_params]
    free_param_names = [param_order[i] for i in free_param_indices]
    print(f"Fixed parameters: {fixed_params}")
    print(f"Free parameters: {free_param_names}")
    if not free_param_names:
         print("Warning: All parameters are fixed. Calculating model directly.")
         # Handle case where nothing needs fitting (left as exercise if needed)
         # You might just calculate Z_model with fixed params and return.

    # 1. Prepare Data (Load, Filter Freq, Apply Median Filter)
    try:
        if data_obj.Zrealimag is None: raise ValueError("Zrealimag data is missing.")
        frequency_raw = data_obj.Zrealimag[:, 0]
        Z_measured_raw = data_obj.Zrealimag[:, 1] + 1j * data_obj.Zrealimag[:, 2]

        frequency = frequency_raw.copy(); Z_measured = Z_measured_raw.copy()
        freq_mask = np.ones_like(frequency, dtype=bool)
        if freq_bounds is not None:
            min_freq, max_freq = freq_bounds
            if min_freq is not None: freq_mask &= (frequency >= min_freq)
            if max_freq is not None: freq_mask &= (frequency <= max_freq)
            frequency = frequency[freq_mask]; Z_measured = Z_measured[freq_mask]
            print(f"Using frequency range: {freq_bounds}. {len(frequency)} points.")
        if len(frequency) == 0: raise ValueError("No data after frequency filtering.")

        kernel_size = med_filt
        if kernel_size is not None and kernel_size > 1:
            if kernel_size % 2 == 0: kernel_size += 1
            if len(frequency) >= kernel_size:
                print(f"Applying median filter (k={kernel_size})...")
                Z_measured = medfilt(Z_measured.real, kernel_size=kernel_size) + 1j * medfilt(Z_measured.imag, kernel_size=kernel_size)
            else: print("Warning: Insufficient points for median filter. Skipping.")

        valid_freq_indices = frequency > 0
        if not np.all(valid_freq_indices):
             frequency = frequency[valid_freq_indices]; Z_measured = Z_measured[valid_freq_indices]
        if len(frequency) == 0: raise ValueError("No valid frequency points > 0.")

    except Exception as e:
        print(f"Error during data preparation: {e}")
        return None, False

    # 2. Define Bounds (DE and LS) for FREE parameters
    # Use provided dicts or defaults
    default_de_bounds = {'R_mem': (1e3, 1e11), 'C_mem': (1e-13, 1e-6), 'C_pad': (1e-13, 1e-6),
                         'Q': (1e-14, 1e-6), 'alpha': (0.5, 1.0), 'R_series': (0, 1e4)}
    default_ls_bounds = {'R_mem': (1e-3, np.inf), 'C_mem': (0, np.inf), 'C_pad': (0, np.inf),
                         'Q': (1e-15, np.inf), 'alpha': (1e-3, 1.0), 'R_series': (0, np.inf)}

    current_de_bounds_dict = de_bounds_dict if de_bounds_dict is not None else default_de_bounds
    current_ls_bounds_dict = ls_bounds_dict if ls_bounds_dict is not None else default_ls_bounds

    try:
        de_bounds_free = [current_de_bounds_dict[name] for name in free_param_names]
        ls_bounds_free = ([current_ls_bounds_dict[name][0] for name in free_param_names],
                          [current_ls_bounds_dict[name][1] for name in free_param_names])
    except KeyError as e:
        print(f"Error: Parameter name '{e}' not found in provided/default bounds dictionaries.")
        return None, False


    # --- Determine Initial Guess for FREE parameters ---
    ls_initial_guess = None
    if use_de:
        # DE doesn't strictly need an initial guess, but LS does later
        print("Running Differential Evolution (Stage 1)...")
        try:
            with warnings.catch_warnings(): # Suppress potential runtime warnings within DE
                 warnings.simplefilter("ignore", category=RuntimeWarning)
                 de_result = differential_evolution(
                    cost_function_de,
                    bounds=de_bounds_free, # Pass bounds for FREE params
                    args=(frequency, Z_measured, model_type_str, param_order, fixed_params_map), # Pass needed args
                    strategy='best1bin', maxiter=de_maxiter, popsize=15, tol=0.01,
                    mutation=(0.5, 1), recombination=0.7, updating='immediate',
                    disp=False # <<< SILENCED DE OUTPUT >>>
                 )

            if de_result.success:
                print("DE finished successfully.")
                ls_initial_guess = de_result.x # Use DE result for LS guess
                print(f"  DE Best Params (Free): {dict(zip(free_param_names, ls_initial_guess))}")
                print(f"  DE Final Cost: {de_result.fun:.4e}")
            else:
                print(f"DE failed: {de_result.message}. Attempting LS with default/provided guess.")
                # Fallback: Try to get guess from dict or auto-guess if DE fails

        except (ValueError, TypeError, IndexError) as e:
             print(f"Error during DE execution: {e}. Check bounds and model function.")
             # Fallback: Try to get guess from dict or auto-guess
        except Exception as e: # Catch other potential errors
             print(f"Unexpected error during DE: {e}")
             # Fallback

    if ls_initial_guess is None: # If DE failed or wasn't used, get guess for LS
         if initial_guess_dict:
             try:
                 ls_initial_guess = [initial_guess_dict[name] for name in free_param_names]
                 print(f"Using provided initial guess for LS: {dict(zip(free_param_names, ls_initial_guess))}")
             except KeyError as e:
                 print(f"Warning: Initial guess missing for '{e}'. Using auto-guess fallback.")
                 ls_initial_guess = None # Trigger auto-guess
         if ls_initial_guess is None: # If no dict or dict failed, auto-guess
             print("Auto-generating initial guess for LS...")
             # Simple auto-guess (adapt as needed)
             r_s_guess = np.median(Z_measured.real[frequency > 0.5 * max(frequency)]) if len(frequency)>1 else 10
             r_mem_guess = np.median(Z_measured.real[frequency < 1.5 * min(frequency)])-r_s_guess if len(frequency)>1 else 1e6
             imag_part = -Z_measured.imag
             peak_idx = np.argmax(imag_part) if len(imag_part)>0 else 0
             f_peak = frequency[peak_idx] if len(frequency)>0 else 1e3
             omega_peak = 2*np.pi*f_peak
             all_guesses = {}
             all_guesses['R_mem'] = max(r_mem_guess, 1e-2)
             all_guesses['R_series'] = max(r_s_guess, 0)
             if use_cpe_model:
                 all_guesses['alpha'] = 0.9
                 all_guesses['Q'] = 1/(omega_peak**all_guesses['alpha'] * all_guesses['R_mem']) if omega_peak >0 and all_guesses['R_mem'] > 0 else 1e-10
             else:
                 c_par_guess = 1/(omega_peak * all_guesses['R_mem']) if omega_peak>0 and all_guesses['R_mem']>0 else 1e-11
                 all_guesses['C_mem'] = max(c_par_guess * 0.9, 1e-15)
                 all_guesses['C_pad'] = max(c_par_guess * 0.1, 1e-15)
             try:
                 ls_initial_guess = [all_guesses[name] for name in free_param_names]
                 print(f"Using auto-generated initial guess for LS: {dict(zip(free_param_names, ls_initial_guess))}")
             except KeyError as e:
                 print(f"Error generating auto-guess for '{e}'. Cannot proceed.")
                 return None, False


    # --- Stage 2: Refinement with Least Squares ---
    print("\nRunning Least Squares Refinement (Stage 2)...")
    fitted_params_dict = None # Initialize
    ls_success = False
    if ls_initial_guess is None:
         print("Error: No valid initial guess available for Least Squares.")
         return None, False

    try:
        ls_result = least_squares(
            residuals_ls_weighted,
            ls_initial_guess,         # Initial guess for FREE params
            args=(frequency, Z_measured, model_type_str, param_order, fixed_params_map), # Pass args
            bounds=ls_bounds_free,    # Bounds for FREE params
            method='trf',
            ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=ls_max_nfev
        )

        # --- Check LS Success and Extract Results ---
        if ls_result.success:
            ls_success = True
            print(f"Least Squares refinement successful!")
            fitted_params_free = ls_result.x
            final_cost = ls_result.cost
            print(f"  LS Final Cost (0.5*sum(sq_resid)): {final_cost:.4e}")

            # Reconstruct full parameter dictionary
            fitted_params_dict = fixed_params.copy() # Start with fixed values
            fitted_params_dict.update(dict(zip(free_param_names, fitted_params_free)))

            print("  Final Fitted Parameters:")
            for name in param_order:
                 print(f"    {name}: {fitted_params_dict[name]:.4e}")

            # --- Calculate final fit curve over FULL frequency range ---
            fitted_params_full_vector = [fitted_params_dict[name] for name in param_order]
            Z_complex_fit_full = unified_circuit_model(fitted_params_full_vector, frequency_raw, model_type_str)
            # Store on data object
            data_obj.Zcomplex_fit = np.column_stack((frequency_raw, Z_complex_fit_full.real, Z_complex_fit_full.imag))
            print("Stored extrapolated fit curve in data_obj.Zcomplex_fit")


            # --- Plotting ---
            if plot_fit:
                fit_label = f"{model_type_str} Fit (LS)"
                if use_de: fit_label = f"{model_type_str} Fit (DE+LS)"

                fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
                scale_factor = 1e6; y_unit = r"M$\Omega$"

                ax[0].plot(frequency_raw, Z_measured_raw.real / scale_factor, 'o', ms=3, color='lightgrey', label='_nolegend_')
                ax[1].plot(frequency_raw, -Z_measured_raw.imag / scale_factor, 'o', ms=3, color='lightgrey', label='_nolegend_')

                # Plot data used for fit (potentially filtered)
                ax[0].plot(frequency, Z_measured.real / scale_factor, 'o', ms=5, label='Measured (used)')
                ax[1].plot(frequency, -Z_measured.imag / scale_factor, 'o', ms=5, label='Measured (used)')

                # Plot fit curve (calculated over full range)
                ax[0].plot(frequency_raw, Z_complex_fit_full.real / scale_factor, '-', lw=2, label=fit_label)
                ax[1].plot(frequency_raw, -Z_complex_fit_full.imag / scale_factor, '-', lw=2, label=fit_label)

                ax[0].set_ylabel(f"Z' ({y_unit})"); ax[0].set_xlabel("Frequency (Hz)"); ax[0].grid(True, which='both'); ax[0].legend(); ax[0].set_title("Real Part")
                ax[0].set_xlim(min(frequency_raw)*0.8, max(frequency_raw)*1.2); ax[0].set_xscale('log')
                ax[1].set_ylabel(f"-Z'' ({y_unit})"); ax[1].set_xlabel("Frequency (Hz)"); ax[1].grid(True, which='both'); ax[1].legend(); ax[1].set_title("Imaginary Part (Negative)")
                ax[1].set_yscale('log'), ax[0].set_yscale('log')
                
                title_note = ""
                if freq_bounds is not None: title_note += f" Freq: {freq_bounds}"
                if med_filt is not None and med_filt > 1: title_note += f" Filter: k={kernel_size}"
                if fixed_params: title_note += f" Fixed: {fixed_params}"
                fig.suptitle(f'Bode Plot - {fit_label} for {data_obj.plot_string}{title_note}', fontsize=12) # Smaller font
                plt.tight_layout(rect=[0, 0.03, 1, 0.93]); plt.show()

        else: # LS failed
            print(f"Least Squares refinement failed: {ls_result.message}")
            # fitted_params_dict remains None

    except ValueError as ve: print(f"ValueError during LS: {ve}"); ls_success = False
    except Exception as e: print(f"Unexpected error during LS: {e}"); ls_success = False

    print(f"--- Fit finished for: {data_obj.plot_string} ---")
    return fitted_params_dict, ls_success