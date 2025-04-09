#import all the libraries needed
from import_dep import *


from scipy.constants import epsilon_0 # Use scipy's constant


# --- Dielectric Model Function (Remains the same) ---
def dielectric_model(params, frequency, n_relaxations=2, C_0=None):
    # ... (No changes needed here) ...
    if C_0 is None or C_0 <= 0: raise ValueError("C_0 must be provided and positive.")
    omega = 2 * np.pi * frequency; epsilon = 1e-18
    param_order = ['eps_inf']; [param_order.extend([f'delta_eps{i}',f'tau{i}']) for i in range(1,n_relaxations+1)]; param_order.append('sigma_dc')
    if len(params) != len(param_order): raise ValueError(f"Incorrect # params for {n_relaxations} relaxations.")
    eps_inf=max(params[0],1.); sigma_dc=max(params[-1],0); debye_terms=0
    for i in range(n_relaxations): d_eps=max(params[1+2*i],epsilon); tau=max(params[2+2*i],epsilon); debye_terms+=d_eps/(1+1j*omega*tau)
    omega_safe=np.maximum(omega,epsilon); cond_term=-1j*(sigma_dc/(omega_safe*epsilon_0)); eps_complex=eps_inf+debye_terms+cond_term
    denom=1j*omega_safe*C_0*eps_complex; denom_safe=np.where(np.abs(denom)<epsilon,epsilon,denom)
    with np.errstate(divide='ignore',invalid='ignore'): Z_model=1/denom_safe
    return np.nan_to_num(Z_model, nan=np.inf)

# --- Weighted Residual Function (Remains the same) ---
def residuals_dielectric_weighted(params, frequency, Z_measured, n_relaxations, C_0):
    # ... (No changes needed here) ...
    Z_model = dielectric_model(params, frequency, n_relaxations, C_0); Z_diff = Z_measured - Z_model
    eps_w = 1e-9; mod_Z = np.abs(Z_measured); weights = 1. / np.maximum(mod_Z, eps_w*np.mean(mod_Z)+eps_w)
    return np.concatenate((Z_diff.real*weights, Z_diff.imag*weights))

# --- Cost Function (Remains the same) ---
def cost_function_dielectric_de(params, frequency, Z_measured, n_relaxations, C_0):
    # ... (No changes needed here) ...
    return np.sum(residuals_dielectric_weighted(params, frequency, Z_measured, n_relaxations, C_0)**2)


# --- Main Fitting Function ---
def fit_impedance_dielectric(
    data_obj,
    n_relaxations=2,
    C_0=None,
    C_pad_to_subtract=None,
    freq_bounds=None,
    med_filt=0,
    fixed_params=None,
    plot_fit=True,
    plot_type='Zabsphi',
    fig_size=(3.5, 2.625), # <<< Default fig size changed to match typical wider Bode plots >>>
                       # This size is for the *entire* figure with 2 subplots
    # Other optimization args with defaults
    use_de=True, 
    de_bounds_dict=None, 
    ls_bounds_dict=None, 
    initial_guess_dict=None,
    de_maxiter=5000, 
    de_popsize=50, 
    de_tol=1e-4,
    ls_max_nfev=3000, 
    ls_ftol=1e-12, 
    ls_xtol=1e-12, 
    ls_gtol=1e-12
    ):
    """
    Fits impedance data to a dielectric relaxation model...
    Args:
        [...]
        fig_size (tuple): Size (width, height) in inches for the ENTIRE figure.
        [...]
    Returns: [... same ...]
    """
    print(f"\n--- Starting Dielectric Fit for: {getattr(data_obj, 'plot_string', 'Unknown Data')} ---")
    # ... (Initial setup and parameter validation - same) ...
    print(f"Using {n_relaxations} Debye relaxation(s)."); # Check C_0
    if C_0 is None or C_0 <= 0: print("Error: C_0 must be provided."); return None, None, None, False
    else: print(f"Using C_0 = {C_0:.4e} F")
    if C_pad_to_subtract is not None and C_pad_to_subtract > 0: print(f"Subtracting parallel C_pad = {C_pad_to_subtract:.4e} F.")
    if fixed_params is None: fixed_params = {}
    else: print(f"Fixed: {fixed_params}")
    if freq_bounds: print(f"Freq range: {freq_bounds}")
    fig, ax = None, None
    param_order = ['eps_inf']; [param_order.extend([f'delta_eps{i}',f'tau{i}']) for i in range(1,n_relaxations+1)]; param_order.append('sigma_dc')
    ALL_POSSIBLE_DEBYE_PARAMS = ['eps_inf','delta_eps1','tau1','delta_eps2','tau2','delta_eps3','tau3','sigma_dc']
    valid_param_names = set(param_order);
    for k in fixed_params:
        if k not in valid_param_names: print(f"Error: Invalid fixed '{k}'. Valid: {param_order}"); return fig,ax,None,False
    fixed_map={param_order.index(k):v for k,v in fixed_params.items()}
    free_idx=[i for i,n in enumerate(param_order) if n not in fixed_params]; free_names=[param_order[i] for i in free_idx]
    print(f"Free params: {free_names}")
    if not free_names: print("All fixed."); return fig,ax,fixed_params,True # Simplified

    # 1. Prepare Data (same logic)
    plot_kernel_size = 0; Z_measured_for_fitting = None
    try: # ... (same data prep: load, freq filter, C_pad subtract, median filter) ...
        if data_obj.Zrealimag is None: raise ValueError("Zrealimag missing.")
        freq_raw=data_obj.Zrealimag[:,0].copy(); Zm_raw=data_obj.Zrealimag[:,1].copy()+1j*data_obj.Zrealimag[:,2].copy()
        freq=freq_raw.copy(); Zm_proc=Zm_raw.copy() # Use Zm_proc for processing steps
        if freq_bounds: min_f,max_f=freq_bounds; mask=np.ones_like(freq,dtype=bool); mask&=(freq>=min_f) if min_f else True; mask&=(freq<=max_f) if max_f else True; freq=freq[mask]; Zm_proc=Zm_proc[mask]; print(f"{len(freq)} pts post-freq filter.")
        if len(freq)==0: raise ValueError("No data post-freq filter.")
        if C_pad_to_subtract and C_pad_to_subtract > 0:
            print("Admittance subtraction..."); omega = 2*np.pi*freq; eps = 1e-18
            with np.errstate(divide='ignore',invalid='ignore'): Ym=1/np.where(np.abs(Zm_proc)<eps,eps,Zm_proc); Yp=1j*omega*C_pad_to_subtract; Yc=Ym-Yp; Ycs=np.where(np.abs(Yc)<eps,eps,Yc); Zm_proc=1/Ycs; Zm_proc=np.nan_to_num(Zm_proc,nan=np.inf); print("Subtracted.")
        ks=med_filt;
        if ks and ks>1:
            if ks%2==0: ks+=1
            if len(freq)>=ks: print(f"Median filter k={ks}"); Zm_proc=medfilt(Zm_proc.real,ks)+1j*medfilt(Zm_proc.imag,ks); plot_kernel_size=ks
            else: print("Warn: Too few pts for median filter.")
        valid_idx=(freq>0)&np.isfinite(Zm_proc);
        if not np.all(valid_idx): print("Warn: Removing non-finite/non-pos freq pts."); freq=freq[valid_idx]; Zm_proc=Zm_proc[valid_idx]
        if len(freq)==0: raise ValueError("No valid data points remain.")
        Z_measured_for_fitting = Zm_proc # Final data for fitting
    except Exception as e: print(f"Error data prep: {e}"); return fig, ax, None, False

    # 2. Define Bounds (DE and LS) for FREE parameters (same logic)
    default_de_bounds = { # Defaults for dielectric params
        'eps_inf':(1.,100.), 'delta_eps1':(1e-3,1e6), 'tau1':(1e-10,1e2), 'delta_eps2':(1e-3,1e5),
        'tau2':(1e-12,1e-2), 'delta_eps3':(1e-3,1e4), 'tau3':(1e-14,1e-5), 'sigma_dc':(0,1e-1)}
    default_ls_bounds = {
        'eps_inf':(1.,np.inf), 'delta_eps1':(0,np.inf), 'tau1':(1e-15,np.inf), 'delta_eps2':(0,np.inf),
        'tau2':(1e-15,np.inf), 'delta_eps3':(0,np.inf), 'tau3':(1e-15,np.inf), 'sigma_dc':(0,np.inf)}
    # ... (Merge user bounds, extract free bounds - same logic) ...
    current_de_bounds=default_de_bounds.copy(); current_ls_bounds=default_ls_bounds.copy()
    if de_bounds_dict: current_de_bounds.update(de_bounds_dict)
    if ls_bounds_dict: current_ls_bounds.update(ls_bounds_dict)
    try: # Check & extract
        for n in free_names:
             if n not in current_de_bounds: raise KeyError(f"DE bound missing '{n}'")
             if n not in current_ls_bounds: raise KeyError(f"LS bound missing '{n}'")
        de_bnds=[current_de_bounds[n] for n in free_names]; ls_bnds=([current_ls_bounds[n][0] for n in free_names],[current_ls_bounds[n][1] for n in free_names])
    except KeyError as e: print(f"Error bounds: {e}"); return fig, ax, None, False

    # 3 & 4: Determine Initial Guess & Run Optimizers (DE + LS)
    ls_initial_guess = None
    args_for_opt = (freq, Z_measured_for_fitting, n_relaxations, C_0) # Use processed Zm
    # ... (DE logic, Fallback guess logic - same as before) ...
    if use_de:
        print("Running DE...");
        try:
            with warnings.catch_warnings(): warnings.simplefilter("ignore",category=RuntimeWarning)
            de_res=differential_evolution(cost_function_dielectric_de, bounds=de_bnds, args=args_for_opt, maxiter=de_maxiter, popsize=de_popsize, tol=de_tol, disp=False)
            if de_res.success: print("DE OK."); ls_initial_guess=de_res.x; print(f"  DE Best: {dict(zip(free_names,ls_initial_guess))}\n  Cost: {de_res.fun:.4e}")
            else: print(f"DE failed: {de_res.message}.")
        except Exception as e: print(f"Error DE: {e}.")
    if ls_initial_guess is None: # Fallback logic
        if initial_guess_dict:
             try: ls_initial_guess=[initial_guess_dict[n] for n in free_names]; print(f"Using provided LS guess: {dict(zip(free_names,ls_initial_guess))}")
             except KeyError as e: print(f"Warn: Guess missing '{e}'. Fallback."); ls_initial_guess=None
        if ls_initial_guess is None: # Midpoint fallback
            print("Using fallback LS guess (midpoint)...")
            try: low_b,high_b=ls_bnds; ls_initial_guess=[(l+h)/2 if np.isfinite(h) else (l*10 if l>0 else 1) for l,h in zip(low_b,high_b)]; print(f"Fallback Guess: {dict(zip(free_names,ls_initial_guess))}")
            except Exception as e_guess: print(f"Error fallback guess: {e_guess}"); return fig, ax, None, False

    # --- Stage 2: Refinement with Least Squares ---
    print("\nRunning LS Refinement...")
    final_fitted_params_dict = None; ls_success = False
    if ls_initial_guess is None: print("Error: No initial guess."); return fig, ax, None, False

    try:
        ls_initial_guess = np.maximum(ls_initial_guess, ls_bnds[0]); ls_initial_guess = np.minimum(ls_initial_guess, ls_bnds[1])
        ls_res = least_squares(residuals_dielectric_weighted, ls_initial_guess, args=args_for_opt, bounds=ls_bnds, method='trf', ftol=ls_ftol, xtol=ls_xtol, gtol=ls_gtol, max_nfev=ls_max_nfev)

        if ls_res.success:
            ls_success = True; print("LS OK!")
            fitted_free = ls_res.x; final_cost = ls_res.cost; print(f"  LS Final Cost: {final_cost:.4e}")
            final_fitted_params_dict = fixed_params.copy(); final_fitted_params_dict.update(dict(zip(free_names, fitted_free)))
            print("  Final Fitted Parameters:"); [print(f"    {name}: {final_fitted_params_dict[name]:.4e}") for name in param_order]

            # --- Store parameters in data_obj.Z_parameters_debye ---
            target_Z_params = {name: None for name in ALL_POSSIBLE_DEBYE_PARAMS}
            for name, value in final_fitted_params_dict.items(): target_Z_params[name] = value
            data_obj.Z_parameters_debye = target_Z_params
            print(f"Stored fitted parameters in data_obj.Z_parameters_debye")

            # --- Calculate final fit curve (adds back C_pad if needed) ---
            Z_complex_fit_full = None
            try:
                fitted_vec = [final_fitted_params_dict[name] for name in param_order]
                if freq_raw is not None and len(freq_raw) > 0:
                     Z_complex_debye_part = dielectric_model(fitted_vec, freq_raw, n_relaxations, C_0)
                     if C_pad_to_subtract and C_pad_to_subtract > 0: # Add back if subtracted
                         print("Adding back subtracted C_pad to final fit curve...")
                         omega_raw=2*np.pi*freq_raw; eps=1e-18
                         with np.errstate(divide='ignore',invalid='ignore'): Ydeb=1/np.where(np.abs(Z_complex_debye_part)<eps,eps,Z_complex_debye_part); Ypad=1j*omega_raw*C_pad_to_subtract; Ytot=Ydeb+Ypad; Ytots=np.where(np.abs(Ytot)<eps,eps,Ytot); Z_complex_fit_full=1/Ytots; Z_complex_fit_full=np.nan_to_num(Z_complex_fit_full,nan=np.inf)
                     else: Z_complex_fit_full = Z_complex_debye_part # Otherwise, model is the full fit
                     # --- Store final Zcomplex_fit in data_obj.Zcomplex_debye_fit ---
                     data_obj.Zcomplex_debye_fit = np.column_stack((freq_raw, Z_complex_fit_full))
                     print("Stored extrapolated fit curve in data_obj.Zcomplex_debye_fit")
                else: print("Warn: Cannot calculate full fit curve.")
            except Exception as e_fc: print(f"Warn: Error calculating final fit curve: {e_fc}")

            # --- Plotting (Using provided fig_size) ---
            if plot_fit and Z_complex_fit_full is not None:
                print("Generating plot...")
                try:
                    
                    double_fig_size = (fig_size[0]*2, fig_size[1])  # Adjusted for two subplots
                    fig, ax = plt.subplots(1, 2, figsize=double_fig_size, constrained_layout=True)

                    fit_label = f"Dielectric Fit ({n_relaxations} relax)" + (" DE+LS" if use_de else " LS")
                    title_note = "" # Build title note
                    if freq_bounds: title_note+=f" Freq:{freq_bounds}";
                    if plot_kernel_size > 1: title_note+=f" Filter:k={plot_kernel_size}"
                    if C_pad_to_subtract: title_note+=f" Csub:{C_pad_to_subtract:.2e}"
                    if fixed_params: title_note+=f" Fixed:{fixed_params}"
                    plot_str=getattr(data_obj,'plot_string','Unk'); plot_str=plot_str or 'Unk'
                    max_z=np.max(np.abs(Zm_raw.real)) if len(Zm_raw)>0 else 1 # Auto-scale units
                    if max_z > 2e6: sf=1e6; yu=r"M$\Omega$"
                    elif max_z > 2e3: sf=1e3; yu=r"k$\Omega$"
                    else: sf=1; yu=r"$\Omega$"

                    # Select data based on plot_type
                    if plot_type=='Zrealimag': y1mr,y2mr=Zm_raw.real/sf,-Zm_raw.imag/sf; y1mf,y2mf=Z_measured_for_fitting.real/sf,-Z_measured_for_fitting.imag/sf; y1f,y2f=Z_complex_fit_full.real/sf,-Z_complex_fit_full.imag/sf; lbl1,lbl2=f"$Z'$ ({yu})",f"$-Z''$ ({yu})"; ys1,ys2='log','linear'
                    elif plot_type=='Zabsphi': y1mr,y2mr=np.abs(Zm_raw)/sf,np.angle(Zm_raw,deg=True); y1mf,y2mf=np.abs(Z_measured_for_fitting)/sf,np.angle(Z_measured_for_fitting,deg=True); y1f,y2f=np.abs(Z_complex_fit_full)/sf,np.angle(Z_complex_fit_full,deg=True); lbl1,lbl2=f"$|Z|$ ({yu})",r"Phase ($^{\circ}$)"; ys1,ys2='log','linear'
                    else: raise ValueError(f"Invalid plot_type: {plot_type}")

                    # Plotting commands
                    ax[0].plot(freq_raw,y1mr,'o',ms=3,c='lightgrey', label='Raw Measured'); ax[1].plot(freq_raw,y2mr,'o',ms=3,c='lightgrey', label='Raw Measured')
                    if C_pad_to_subtract or med_filt or freq_bounds: # Show processed data if different
                        ax[0].plot(freq,y1mf,'o',ms=5, label='Data Used for Fit'); ax[1].plot(freq,y2mf,'o',ms=5, label='Data Used for Fit')
                    else: # If no processing, label the raw data as used
                        ax[0].plot([],[], 'o', ms=5, label='Measured (used)') # Add dummy for legend if needed
                        ax[1].plot([],[], 'o', ms=5, label='Measured (used)')

                    ax[0].plot(freq_raw,y1f,'-',lw=2,label=fit_label); ax[1].plot(freq_raw,y2f,'-',lw=2,label=fit_label)

                    # Formatting
                    ax[0].set_ylabel(lbl1); ax[0].set_xlabel("Frequency (Hz)"); ax[0].grid(True,which='both'); ax[0].legend(); ax[0].set_title("Plot 1")
                    ax[0].set_xlim(min(freq_raw)*0.8, max(freq_raw)*1.2); ax[0].set_xscale('log'); ax[0].set_yscale(ys1)
                    ax[1].set_ylabel(lbl2); ax[1].set_xlabel("Frequency (Hz)"); ax[1].grid(True,which='both'); ax[1].legend(); ax[1].set_title("Plot 2")
                    ax[1].set_yscale(ys2)
                    fig.suptitle(f'Bode ({plot_type}) - {fit_label} for {plot_str}{title_note}', fontsize=12);
                    # plt.tight_layout(rect=[0,0.03,1,0.93]); # Let constrained_layout handle spacing
                    plt.show()
                except Exception as e_plot: print(f"Error plot: {e_plot}"); plt.close(fig if 'fig' in locals() else None)
            elif plot_fit: print("Skipping plot: Failed calc.")

        else: # LS failed
            print(f"LS failed: {ls_res.message}"); ls_success = False

    except ValueError as ve: print(f"ValueError LS stage: {ve}"); ls_success = False
    except Exception as e: print(f"Unexpected error LS stage: {e}"); ls_success = False

    print(f"--- Fit finished for: {getattr(data_obj, 'plot_string', 'Unknown Data')} ---")
    return fig, ax