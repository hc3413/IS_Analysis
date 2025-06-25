#import all the libraries needed
from import_dep import *



def generate_colormaps_and_normalizers(data_in, c_bar):
        """
        Generate colormap and normalizer based on temperature or DC offset.
        """
        if c_bar == 1:  # Colorbar for temperature
            values = [m.Temperature for m in data_in if m.Temperature is not None]
        elif c_bar == 2:  # Colorbar for DC offset
            values = [m.DC_offset for m in data_in if m.DC_offset is not None]
        elif c_bar == 3: # Colorbar for V_rms
            values = [m.V_rms for m in data_in if m.V_rms is not None]
        else:
            return None, None, None, None

        min_val, max_val = min(values), max(values)
        norm = Normalize(vmin=min_val, vmax=max_val)
        cmap = plt.get_cmap('coolwarm')
        return cmap, norm, min_val, max_val
        
def IS_plot(
    data_in: list,
    d_type: str,
    x_lim: tuple = None,
    y_lim_left: tuple = None, # limit for the left plot y axis
    y_lim_right: tuple = None, # limit for the right plot y axis
    sort_data: bool = True, # Order the data by temperature and DC offset
    fig_size: tuple = (3.5, 2.625), # Base size of the figure
    c_bar: int = 0, # 0 = no colorbar, 1 = temperature, 2 = DC_offset
    med_filt: int = 0, # 0 = no median filter, x = size of the median filter
    force_key: bool = False, # Force a key to be created even if c_bar is activated
    show_key: bool = True, # Show key/legend - changed default to True to show parameters
    freq_lim: tuple = None, # Frequency limits for limiting data in colecole and modmod plot
    fitting: bool = False, # Plot the fitted data along with the measured data
):
    '''Plotting function for impedance data using constrained_layout for consistency.

    data_in: list - the list of Measurement class data to plot
    d_type: str - the type of data to plot: 'Zabsphi', 'Zrealimag', 'permittivity', 'tandelta', 'modulus'
    fig_size: tuple - Desired base figure size. Constrained layout might adjust slightly.
    c_bar: int - 0 = no colorbar, 1 = temperature, 2 = DC_offset
    '''

    if not data_in:
        print("No data to plot")
        return None, None

    # --- Create Figure and Axes using constrained_layout ---
    if d_type in ['colecole', 'modmod']:
        fig, ax = plt.subplots(1, 1, figsize=fig_size, constrained_layout=True)
    else:
        double_fig_size = (fig_size[0]*2, fig_size[1])  # Adjusted for two subplots
        fig, ax = plt.subplots(1, 2, figsize=double_fig_size, constrained_layout=True)

    # Ensure ax is always iterable
    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])

    # Titles and labels (your existing dict is fine)
    titles = {
        'Zabsphi': (r"$|Z|\,(\Omega)$", r"Phase ($^{\circ}$)"),
        'Zrealimag': (r"$Z'\,(\Omega)$", r"$-Z''\,(\Omega)$"),
        'permittivity': (r"$\varepsilon'$", r"$\varepsilon''$"),
        'tandelta': (r"$\sigma$ (S/m)", r"$\tan\delta$"), # Check if sigma is correct for left axis here
        'modulus': (r"$M'$", r"$M''$"),
        'colecole': (r"$Z'$", r"$-Z''$"), # Cole-Cole plot
        'modmod': (r"$M'$", r"$-M''$"), # Modulus plot
    }

    if d_type not in titles:
        print(f"Error: Invalid data type '{d_type}'. Valid types are: {list(titles.keys())}")
        plt.close(fig) # Close the empty figure
        return None, None

    ylabels = titles[d_type]

    # Default color cycle if no colorbar
    if c_bar == 0:
        # Use the default prop_cycle defined in your style
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

    # Define linestyles for different sublists
    linestyles = ['-', ':','--', '-.', ]

    # Check if data_in is a list of lists
    if isinstance(data_in[0], list):
        data_groups = data_in  # Treat each sublist as a group
    else:
        data_groups = [data_in]  # Wrap single list into a group for uniform handling

    # --- Plotting Loop ---
    plotted_lines_left = []  # For legend handles
    plotted_lines_right = []
    group_legend_entries = []  # To store legend entries for groups

    for group_idx, group in enumerate(data_groups):
        linestyle = linestyles[group_idx % len(linestyles)]  # Cycle through linestyles

        # Sort data within the group if enabled
        if sort_data:
            group.sort(key=lambda m: (
                0 if m.Temperature is not None else 1,
                round(m.Temperature, -1) if m.Temperature is not None else float('inf'),
                0 if m.DC_offset is not None else 1,
                m.DC_offset if m.DC_offset is not None else float('inf')
            ))

        # Generate colormap and normalizer for the group if colorbar is enabled
        cmap, norm, min_val, max_val = generate_colormaps_and_normalizers(group, c_bar)

        # Add a legend entry for the group using the run number of the first item
        group_legend_entries.append((f"{group[0].plot_string}", linestyle))
        
        

        for i, measurement in enumerate(group):
            plot_string = measurement.plot_string  # Label for legend

            # --- Data Extraction ---
            if d_type == 'Zabsphi':
                data = measurement.Zabsphi
                x, y1, y2 = data[:, 0], data[:, 1], data[:, 2]
                ax[0].set_yscale('log') # Apply scale here
            elif d_type == 'Zrealimag':
                data = measurement.Zrealimag
                x, y1, y2 = data[:, 0], data[:, 1], -data[:, 2]
                ax[0].set_yscale('log') # Apply scale if needed
            elif d_type == 'permittivity':
                data = measurement.permittivity
                x, y1, y2 = data[:, 0], data[:, 1], data[:, 2]
                # Optional: Apply scales if needed (e.g., log for permittivity)
                # ax[0].set_yscale('log')
                # ax[1].set_yscale('log')
            elif d_type == 'tandelta':
                cond_data = measurement.conductivity
                tan_delta_data = measurement.tandelta
                x, y1, y2 = cond_data[:, 0], cond_data[:, 1], tan_delta_data[:, 1]
                ax[0].set_yscale('log') # Conductivity often log
                # ax[1].set_yscale('log') # Tan delta sometimes log
            elif d_type == 'modulus':
                data = measurement.modulus
                x, y1, y2 = data[:, 0], data[:, 1], data[:, 2]
                # Optional: Apply scales if needed
                # ax[0].set_yscale('log')
                # ax[1].set_yscale('log')
                
            elif d_type == 'colecole':
                data = measurement.Zrealimag
                x, y1, y2 = data[:, 1], -data[:, 2], data[:, 0] #passing frequency as dummy entry for y2 to enable filtering
            
            elif d_type == 'modmod':
                data = measurement.modulus
                x, y1, y2 = data[:, 1], -data[:, 2], data[:, 0] #passing frequency as dummy entry for y2 to enable filtering
                
            else: # Should not happen due to check above, but good practice
                print(f"Error: Invalid data type '{d_type}'. Valid types are: {list(titles.keys())}")
                continue

            # --- Apply Median Filter if Enabled ---
            if med_filt > 0:
                y1 = medfilt(y1, kernel_size=med_filt)
                y2 = medfilt(y2, kernel_size=med_filt)

            # --- Masking for limits ---
            valid_mask = ~np.isnan(x) & ~np.isnan(y1) & ~np.isnan(y2) # Basic check
            x, y1, y2 = x[valid_mask], y1[valid_mask], y2[valid_mask]

            if x_lim:
                x_mask = (x >= x_lim[0]) & (x <= x_lim[1])
                x, y1, y2 = x[x_mask], y1[x_mask], y2[x_mask]

            x1, x2 = x.copy(), x.copy() # Keep separate copies for y-masking

            if y_lim_left:
                y1_mask = (y1 >= y_lim_left[0]) & (y1 <= y_lim_left[1])
                x1_masked, y1_masked = x1[y1_mask], y1[y1_mask]
            else:
                x1_masked, y1_masked = x1, y1 # Use all data if no limit

            if y_lim_right:
                y2_mask = (y2 >= y_lim_right[0]) & (y2 <= y_lim_right[1])
                x2_masked, y2_masked = x2[y2_mask], y2[y2_mask]
            else:
                x2_masked, y2_masked = x2, y2 # Use all data if no limit
                
            # --- Masking for frequency limits in colecole and modmod plots ---
            if d_type in ['colecole', 'modmod'] and freq_lim:
                freq_mask = (y2 >= freq_lim[0]) & (y2 <= freq_lim[1])
                x1_masked, y1_masked = x1[freq_mask], y1[freq_mask]
            elif freq_lim:
                freq_mask = (x2 >= freq_lim[0]) & (x2 <= freq_lim[1])
                x1_masked,x2_masked, y1_masked, y2_masked = x1[freq_mask], x2[freq_mask], y1[freq_mask], y2[freq_mask]


            # Skip plotting if no data remains after masking
            if x1_masked.size == 0 and x2_masked.size == 0:
                print(f"Warning: No data points left for {plot_string} after applying limits.")
                continue

            # --- Determine Color ---
            if c_bar == 1 and measurement.Temperature is not None:
                color = cmap(norm(measurement.Temperature))
            elif c_bar == 2 and measurement.DC_offset is not None:
                color = cmap(norm(measurement.DC_offset))
            elif c_bar == 3 and measurement.V_rms is not None:
                color = cmap(norm(measurement.V_rms))
            else:
                # Use default color cycle if no colorbar or value is None
                color = colors[i % len(colors)] # Cycle through default colors

            # --- Plot ---

            # Plot for Cole-Cole and ModMod plots
            if d_type in ['colecole', 'modmod']:
                line1 = ax[0].plot(x1_masked, y1_masked, label=plot_string, color=color, linestyle=linestyle)
                plotted_lines_left.extend(line1)  # Store handle for legend
            
            # Standard plots
            else:
                line1 = ax[0].semilogx(x1_masked, y1_masked, label=plot_string, color=color, linestyle=linestyle)
                plotted_lines_left.extend(line1)  # Store handle for legend
            
                line2 = ax[1].semilogx(x2_masked, y2_masked, label=plot_string, color=color, linestyle=linestyle)
                plotted_lines_right.extend(line2)  # Store handle for legend


    # --- Axis Labels and Limits ---
    if d_type in ['colecole', 'modmod']:
        ax[0].set_xlabel(ylabels[0])
        ax[0].set_ylabel(ylabels[1])
    else:
        ax[0].set_xlabel('Frequency (Hz)')
        ax[1].set_xlabel('Frequency (Hz)')
        ax[0].set_ylabel(ylabels[0])
        ax[1].set_ylabel(ylabels[1])

    # Apply limits AFTER plotting and AFTER setting scale
    if x_lim:
        ax[0].set_xlim(x_lim)
        if len(ax) > 1:
            ax[1].set_xlim(x_lim)
    if y_lim_left:
        ax[0].set_ylim(y_lim_left)
    if y_lim_right and len(ax) > 1:
        ax[1].set_ylim(y_lim_right)


    # --- Add Colorbar (if applicable) ---
    if c_bar and cmap is not None: # Check if cmap was successfully created
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # Add colorbar, letting constrained_layout place it.
        # Adjust 'shrink' and 'aspect' as needed for aesthetics.
        # `location='right'` is default for vertical colorbar with subplots(1,N)
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8, aspect=20) # ax=ax attaches to both

        # Set colorbar label and ticks (similar to your old add_colorbar)
        cbar.set_ticks([min_val, max_val]) # Set major ticks explicitly
        if min_val == max_val: # Handle single value case for labels
             if c_bar == 1:
                 cbar.set_ticklabels([f'{min_val:.1f} K', f'single value'])
                 cbar.set_label("Temperature (K)")
             elif c_bar == 2:
                 cbar.set_ticklabels([f'{min_val:.1f} V', f'single value'])
                 cbar.set_label("DC Offset (V)")
             elif c_bar == 3:
                 cbar.set_ticklabels([f'{min_val:.1f} V_rms', f'single value'])
                 cbar.set_label("V_rms (V)")
        else:
             if c_bar == 1:
                 cbar.set_ticklabels([f'{min_val:.1f}', f'{max_val:.1f}']) # Just numbers, label indicates units
                 cbar.set_label("Temperature (K)")
             elif c_bar == 2:
                 cbar.set_ticklabels([f'{min_val:.1f}', f'{max_val:.1f}']) # Just numbers, label indicates units
                 cbar.set_label("DC Offset (V)")
             elif c_bar == 3:
                 cbar.set_ticklabels([f'{min_val:.1f}', f'{max_val:.1f}']) # Just numbers, label indicates units
                 cbar.set_label("V_rms (V)")

        cbar.minorticks_off()
        cbar.outline.set_linewidth(0.5)
        
        
    # --- Legends ---
    if show_key:  # Show legend if enabled
        if c_bar != 0 and not force_key and len(data_groups)>1 :  # Add group legend when colorbar is enabled and force_key is disabled
            group_handles = [
                plt.Line2D([0], [0], color='black', linestyle=linestyle, label=label)
                for label, linestyle in group_legend_entries
            ]
            ax[0].legend(handles=group_handles)

        elif c_bar == 0 or force_key:  # Default legend behavior
            handles, labels = ax[0].get_legend_handles_labels()
            ax[0].legend(handles, labels)  # Customize further if needed

            if len(ax) > 1:
                handles, labels = ax[1].get_legend_handles_labels()
                ax[1].legend(handles, labels)

    # Layout and show (constrained_layout handles adjustments before showing)
    plt.show()

    return fig, ax



def IS_plot_fit(
    data_in: list,
    d_type: str,
    x_lim: tuple = None,
    y_lim_left: tuple = None, # limit for the left plot y axis
    y_lim_right: tuple = None, # limit for the right plot y axis
    sort_data: bool = True, # Order the data by temperature and DC offset
    fig_size: tuple = (3.5, 2.625), # Base size of the figure
    med_filt: int = 0, # 0 = no median filter, x = size of the median filter
    show_key: bool = True, # Show key/legend - changed default to True to show parameters
    freq_lim: tuple = None, # Frequency limits for limiting data in colecole and modmod plot
    fit_type: str = 'RC', # 'RC' for main fit, 'debye' for Debye fit
):
    '''Plotting function for impedance data with fitted model curves using constrained_layout.
    
    This function always shows measured data as points and fitted data as lines.

    data_in: list - the list of Measurement class data to plot
    d_type: str - the type of data to plot: 'Zabsphi', 'Zrealimag', 'permittivity', 'tandelta', 'modulus'
    fig_size: tuple - Desired base figure size. Constrained layout might adjust slightly.
    fit_type: str - 'RC' (default) for main fit, 'debye' for Debye fit
    '''

    if not data_in:
        print("No data to plot")
        return None, None

    # --- Create Figure and Axes using constrained_layout ---
    if d_type in ['colecole', 'modmod']:
        fig, ax = plt.subplots(1, 1, figsize=fig_size, constrained_layout=True)
    else:
        double_fig_size = (fig_size[0]*2, fig_size[1])  # Adjusted for two subplots
        fig, ax = plt.subplots(1, 2, figsize=double_fig_size, constrained_layout=True)

    # Ensure ax is always iterable
    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])

    # Titles and labels
    titles = {
        'Zabsphi': (r"$|Z|\\,(\\Omega)$", r"Phase ($^{\\circ}$)"),
        'Zrealimag': (r"$Z'\\,(\\Omega)$", r"$-Z''\\,(\\Omega)$"),
        'permittivity': (r"$\\varepsilon'$", r"$\\varepsilon''$"),
        'tandelta': (r"$\\sigma$ (S/m)", r"$\\tan\\delta$"), 
        'modulus': (r"$M'$", r"$M''$"),
        'colecole': (r"$Z'$", r"$-Z''$"), 
        'modmod': (r"$M'$", r"$-M''$"), 
    }

    if d_type not in titles:
        print(f"Error: Invalid data type '{d_type}'. Valid types are: {list(titles.keys())}")
        plt.close(fig)
        return None, None

    ylabels = titles[d_type]

    # Default color cycle for data
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    # Generate a list of colormaps of single colors for use with multiple groups
    color_maps = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples', 'Greys']
    color_map_iterator = itertools.cycle(color_maps) #makes an iterator object that can be cyled through with next() to get the next color map

    # Define linestyles for different sublists
    linestyles = ['-', ':','--', '-.']

    # Check if data_in is a list of lists
    if isinstance(data_in[0], list):
        data_groups = data_in  # Treat each sublist as a group
    else:
        data_groups = [data_in]  # Wrap single list into a group for uniform handling

    # --- Plotting Loop ---
    plotted_lines_left = []  # For legend handles
    plotted_lines_right = []
    group_legend_entries = []  # To store legend entries for groups

    for group_idx, group in enumerate(data_groups):
        linestyle = linestyles[group_idx % len(linestyles)]  # Cycle through linestyles

        # Sort data within the group if enabled
        if sort_data:
            group.sort(key=lambda m: (
                0 if m.Temperature is not None else 1,
                round(m.Temperature, -1) if m.Temperature is not None else float('inf'),
                0 if m.DC_offset is not None else 1,
                m.DC_offset if m.DC_offset is not None else float('inf')
            ))
            
        # If passed a group, then use a new colormap for each group
        if len(data_groups) > 1:
            # Generate a list of colors from the colormap
            color_obj = next(color_map_iterator) #get the first color map to pass as a plotting argument
            
            num_colors = len(group) # set number of colors to number of devices within the subset
            colors = plt.get_cmap(color_obj)(np.linspace(0.2,0.8,num_colors))
            

        # Add a legend entry for the group using the run number of the first item
        group_legend_entries.append((f"{group[0].plot_string}", linestyle))

        for i, measurement in enumerate(group):
            plot_string = measurement.plot_string  # Original label
            has_fit_data = hasattr(measurement, 'Zcomplex_fit') and measurement.Zcomplex_fit is not None
            
            if not has_fit_data:
                print(f"Warning: No fitted data found for {plot_string}. Skipping.")
                continue
                
            # Create formatted parameter string for the plot label
            param_string = ""
            if hasattr(measurement, 'Z_parameters') and measurement.Z_parameters is not None:
                params = measurement.Z_parameters
                DC_offset = measurement.DC_offset
                # Format only non-None parameters with scientific notation for clarity
                param_entries = []
                
                # Handle RC circuit parameters
                if params.get('R_mem1') is not None:
                    param_entries.append(f"R1={params['R_mem1']:.1e}")
                if params.get('C_mem1') is not None:
                    param_entries.append(f"C1={params['C_mem1']:.2e}")
                
                # Handle RC2 circuit parameters
                if params.get('R_mem2') is not None:
                    param_entries.append(f"R2={params['R_mem2']:.1e}")
                if params.get('C_mem2') is not None:
                    param_entries.append(f"C2={params['C_mem2']:.2e}")
                
                # Handle CPE parameters
                if params.get('Q1') is not None:
                    param_entries.append(f"Q1={params['Q1']:.2e}")
                if params.get('alpha1') is not None:
                    param_entries.append(f"α1={params['alpha1']:.2f}")
                if params.get('Q2') is not None:
                    param_entries.append(f"Q2={params['Q2']:.2e}")
                if params.get('alpha2') is not None:
                    param_entries.append(f"α2={params['alpha2']:.2f}")
                
                # Handle common parameters
                if params.get('R_series') is not None:
                    param_entries.append(f"Rs={params['R_series']:.1e}")
                #if params.get('C_pad') is not None:
                    #param_entries.append(f"Cp={params['C_pad']:.1e}")
                    
                if DC_offset is not None:
                    param_entries.append(f"DC={DC_offset:.1f}")
                    
                param_string = ", ".join(param_entries)
            
            # Create the new labels with run number and parameters
            run_num = f"run={measurement.run_number}" if hasattr(measurement, 'run_number') else "unknown"
            fit_label = f"{run_num}: {param_string}"

            # --- Data Extraction for Measured Data ---
            if d_type == 'Zabsphi':
                data = measurement.Zabsphi
                x, y1, y2 = data[:, 0], data[:, 1], data[:, 2]
                ax[0].set_yscale('log')
                fit_data = getattr(measurement, 'Zabsphi_fit' if fit_type=='RC' else 'Zabsphi_fit_debye', None)
                if fit_data is not None:
                    fit_x, fit_y1, fit_y2 = fit_data[:, 0], fit_data[:, 1], fit_data[:, 2]
            elif d_type == 'Zrealimag':
                data = measurement.Zrealimag
                x, y1, y2 = data[:, 0], data[:, 1], -data[:, 2]
                ax[0].set_yscale('log')
                fit_data = getattr(measurement, 'Zrealimag_fit' if fit_type=='RC' else 'Zrealimag_fit_debye', None)
                if fit_data is not None:
                    fit_x, fit_y1, fit_y2 = fit_data[:, 0], fit_data[:, 1], -fit_data[:, 2]
            elif d_type == 'permittivity':
                data = measurement.permittivity
                x, y1, y2 = data[:, 0], data[:, 1], data[:, 2]
                fit_data = getattr(measurement, 'permittivity_fit' if fit_type=='RC' else 'permittivity_fit_debye', None)
                if fit_data is not None:
                    fit_x, fit_y1, fit_y2 = fit_data[:, 0], fit_data[:, 1], fit_data[:, 2]
            elif d_type == 'tandelta':
                cond_data = measurement.conductivity
                tan_delta_data = measurement.tandelta
                x, y1, y2 = cond_data[:, 0], cond_data[:, 1], tan_delta_data[:, 1]
                ax[0].set_yscale('log')
                fit_data = getattr(measurement, 'tandelta_fit' if fit_type=='RC' else 'tandelta_fit_debye', None)
                if fit_data is not None:
                    fit_x, fit_y1, fit_y2 = fit_data[:, 0], fit_data[:, 1], fit_data[:, 2]
            elif d_type == 'modulus':
                data = measurement.modulus
                x, y1, y2 = data[:, 0], data[:, 1], data[:, 2]
                fit_data = getattr(measurement, 'modulus_fit' if fit_type=='RC' else 'modulus_fit_debye', None)
                if fit_data is not None:
                    fit_x, fit_y1, fit_y2 = fit_data[:, 0], fit_data[:, 1], fit_data[:, 2]
            elif d_type == 'colecole':
                data = measurement.Zrealimag
                x, y1, y2 = data[:, 1], -data[:, 2], data[:, 0]
                fit_data = getattr(measurement, 'Zrealimag_fit' if fit_type=='RC' else 'Zrealimag_fit_debye', None)
                if fit_data is not None:
                    fit_x, fit_y1, fit_y2 = fit_data[:, 1], -fit_data[:, 2], fit_data[:, 0]
            elif d_type == 'modmod':
                data = measurement.modulus
                x, y1, y2 = data[:, 1], -data[:, 2], data[:, 0]
                fit_data = getattr(measurement, 'modulus_fit' if fit_type=='RC' else 'modulus_fit_debye', None)
                if fit_data is not None:
                    fit_x, fit_y1, fit_y2 = fit_data[:, 1], -fit_data[:, 2], fit_data[:, 0]
            else:
                fit_data = None
            if fit_data is None:
                print(f"Warning: No fitted data found for {plot_string} (fit_type={fit_type}, d_type={d_type}). Skipping fit plot.")
                continue
            
            # --- Apply Median Filter (only to measured data) ---
            if med_filt > 0:
                y1 = medfilt(y1, kernel_size=med_filt)
                y2 = medfilt(y2, kernel_size=med_filt)

            # --- Masking for limits - measured data ---
            # Remove NaN values
            valid_mask = ~np.isnan(x) & ~np.isnan(y1) & ~np.isnan(y2)
            x, y1, y2 = x[valid_mask], y1[valid_mask], y2[valid_mask]

            # X-axis limits
            if x_lim:
                x_mask = (x >= x_lim[0]) & (x <= x_lim[1])
                x, y1, y2 = x[x_mask], y1[x_mask], y2[x_mask]
                
                # Apply same limits to fitted data
                fit_x_mask = (fit_x >= x_lim[0]) & (fit_x <= x_lim[1])
                fit_x = fit_x[fit_x_mask]
                fit_y1 = fit_y1[fit_x_mask]
                fit_y2 = fit_y2[fit_x_mask]
                
            # Create separate copies for y-axis masking
            x1, x2 = x.copy(), x.copy()
            fit_x1, fit_x2 = fit_x.copy(), fit_x.copy()

            # Y-axis left limits
            if y_lim_left:
                y1_mask = (y1 >= y_lim_left[0]) & (y1 <= y_lim_left[1])
                x1_masked, y1_masked = x1[y1_mask], y1[y1_mask]
                
                fit_y1_mask = (fit_y1 >= y_lim_left[0]) & (fit_y1 <= y_lim_left[1])
                fit_x1_masked, fit_y1_masked = fit_x1[fit_y1_mask], fit_y1[fit_y1_mask]
            else:
                x1_masked, y1_masked = x1, y1
                fit_x1_masked, fit_y1_masked = fit_x1, fit_y1

            # Y-axis right limits
            if y_lim_right:
                y2_mask = (y2 >= y_lim_right[0]) & (y2 <= y_lim_right[1])
                x2_masked, y2_masked = x2[y2_mask], y2[y2_mask]
                
                if d_type not in ['colecole', 'modmod']:
                    fit_y2_mask = (fit_y2 >= y_lim_right[0]) & (fit_y2 <= y_lim_right[1])
                    fit_x2_masked, fit_y2_masked = fit_x2[fit_y2_mask], fit_y2[fit_y2_mask]
            else:
                x2_masked, y2_masked = x2, y2
                fit_x2_masked, fit_y2_masked = fit_x2, fit_y2
                
            # Frequency limits for Cole-Cole and ModMod plots
            if d_type in ['colecole', 'modmod'] and freq_lim:
                freq_mask = (y2 >= freq_lim[0]) & (y2 <= freq_lim[1])
                x1_masked, y1_masked = x1[freq_mask], y1[freq_mask]
                
                fit_freq_mask = (fit_x >= freq_lim[0]) & (fit_x <= freq_lim[1])
                fit_x1_masked, fit_y1_masked = fit_x1[fit_freq_mask], fit_y1[fit_freq_mask]
                fit_y2_masked = fit_y2[fit_freq_mask]
            elif freq_lim:
                freq_mask = (x2 >= freq_lim[0]) & (x2 <= freq_lim[1])
                x1_masked, x2_masked, y1_masked, y2_masked = x1[freq_mask], x2[freq_mask], y1[freq_mask], y2[freq_mask]
                
                fit_freq_mask = (fit_x >= freq_lim[0]) & (fit_x <= freq_lim[1])
                fit_x1_masked, fit_x2_masked = fit_x1[fit_freq_mask], fit_x2[fit_freq_mask]
                fit_y1_masked, fit_y2_masked = fit_y1[fit_freq_mask], fit_y2[fit_freq_mask]

            # Skip plotting if no data remains after masking
            if x1_masked.size == 0 and x2_masked.size == 0:
                print(f"Warning: No data points left for {plot_string} after applying limits.")
                continue

            # --- Determine Color ---
            color = colors[i % len(colors)]  # Cycle through colors

            
            # --- Plot Data ---
            # Cole-Cole and ModMod plots
            if d_type in ['colecole', 'modmod']:
                # Measured data as circles
                line1 = ax[0].plot(x1_masked, y1_masked, 'o', ms=4, color=color)
                plotted_lines_left.extend(line1)
                
                # Fitted data as solid lines
                ax[0].plot(fit_y1_masked, fit_y2_masked, '-', lw=1.5,
                          label=fit_label, color=color)
            
            # Standard plots (Bode plots, etc.)
            else:
                # Measured data as circles
                line1 = ax[0].semilogx(x1_masked, y1_masked, 'o', ms=3, alpha = 0.2, color=color)
                plotted_lines_left.extend(line1)
                
                line2 = ax[1].semilogx(x2_masked, y2_masked, 'o', ms=3, alpha = 0.2, color=color)
                plotted_lines_right.extend(line2)
                
                # Fitted data as solid lines
                ax[0].semilogx(fit_x1_masked, fit_y1_masked, '--', lw=1.5, alpha = 1,
                              label=fit_label, color=color)
                ax[1].semilogx(fit_x2_masked, fit_y2_masked, '--', lw=1.5, alpha = 1,
                              label=fit_label, color=color)

    # --- Axis Labels and Limits ---
    if d_type in ['colecole', 'modmod']:
        ax[0].set_xlabel(ylabels[0])
        ax[0].set_ylabel(ylabels[1])
        ax[0].axis('equal')
    else:
        ax[0].set_xlabel('Frequency (Hz)')
        ax[1].set_xlabel('Frequency (Hz)')
        ax[0].set_ylabel(ylabels[0])
        ax[1].set_ylabel(ylabels[1])

    # Apply limits AFTER plotting and AFTER setting scale
    if x_lim:
        ax[0].set_xlim(x_lim)
        if len(ax) > 1:
            ax[1].set_xlim(x_lim)
    if y_lim_left:
        ax[0].set_ylim(y_lim_left)
    if y_lim_right and len(ax) > 1:
        ax[1].set_ylim(y_lim_right)
    if show_key:
        handles, labels = ax[0].get_legend_handles_labels()
        ax[0].legend(handles, labels)
        if len(ax) > 1:
            handles, labels = ax[1].get_legend_handles_labels()
            ax[1].legend(handles, labels)
    plt.show()
    return fig, ax

def run_to_dict(data_in):
    '''
    loop through the imported data and put all measurements with the same run number into a list
    With each list containing the data for a single run put into a dictionary
    '''
    data_dict = {}
    for measurement in data_in:
        # if the run number is not in the dictionary, create a new list for it
        if measurement.run_number not in data_dict:
            data_dict[measurement.run_number] = []
        # append the measurement to the list for that run number
        data_dict[measurement.run_number].append(measurement)
    return data_dict


def update_plot_string(data_in, export_data=False, plot_labels = None):
    '''Update the plot_str variable for each file in the tuple.
      update only the first measurement in the list which is the one actually outputted in the plot'''
    if export_data:
        for i, list in enumerate(data_in, start=0):
            
            # Manual input if no plot_labels provided
            if plot_labels is None:
                new_plot_str = input(f"Enter new plot string for {list[0].file_name} (current: {list[0].plot_string}): ")
                list[0].plot_string = new_plot_str if new_plot_str else list[0].plot_string
                print(f"New plot string for {list[0].file_name}: {list[0].plot_string}")
            
            # Automatic input if plot_labels provided
            else:
                list[0].plot_string = plot_labels[i]
                
        return data_in
    
    else:
        return data_in # Do nothing and exit the function
    
def extract_single_dc(data_in, DC_val = 0):
    '''Extract a single DC offset from the data and return it
    Do this for each list within the combined list for plotting'''
    dat_filt = []
    for run in data_in:
        run_filt = []
        for measurement in run:
            if round(measurement.DC_offset, 2) == round(DC_val, 2):
                # Append the measurement to the list
                run_filt.append(measurement)
        # Append the filtered run to the main list
        dat_filt.append(run_filt)
    return dat_filt












### Legacy functions
# def add_colorbar(fig, cax, cmap, norm, min_val, max_val, c_bar, fig_size):
#     """
#     Add a colorbar to the figure for both axes.
#     """
    
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     cbar = plt.colorbar(sm, cax=cax)
#     cbar.set_ticks([min_val, max_val])
    
#     if c_bar == 1:
#         cbar.set_label("Temperature")
#         cbar.set_ticklabels([f'{min_val:.1f} K', f'{max_val:.1f} K']) # Temperature
#     else:
#         cbar.set_label("DC Offset")
#         cbar.set_ticklabels([f'{min_val:.1f} V', f'{max_val:.1f} V']) # DC offset
        
#     cbar.minorticks_off()  # Remove minor ticks
#     cbar.outline.set_linewidth(0.5)

#     # Adjust colorbar position and size based on figure size
#     if fig_size == [3.5, 2.625]:
#         height_scale = 0.8
#         pos = cax.get_position()
#         new_height = pos.height * height_scale
#         new_y0 = pos.y0 + (pos.height - new_height) / 2  # Center the colorbar vertically
#         cax.set_position([
#             pos.x0 + 0.03,  # Adjust x0 to move the colorbar to the right
#             new_y0,  # Center the colorbar vertically
#             pos.width * 0.3,  # Adjust width to shrink the colorbar
#             new_height  # Adjust height to shrink the colorbar
#         ])
#     else:
#         height_scale = 0.8
#         pos = cax.get_position()
#         new_height = pos.height * height_scale
#         new_y0 = pos.y0 + (pos.height - new_height) / 2  # Center the colorbar vertically
#         cax.set_position([
#             pos.x0 + 0.03,  # Adjust x0 to move the colorbar to the right
#             new_y0,  # Center the colorbar vertically
#             pos.width * 0.3,  # Adjust width to shrink the colorbar
#             new_height  # Adjust height to shrink the colorbar
#         ])
