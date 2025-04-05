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

    if sort_data:
        # Sort data (your existing logic is fine)
        data_in.sort(key=lambda m: (
            0 if m.Temperature is not None else 1,
            round(m.Temperature, -1) if m.Temperature is not None else float('inf'),
            0 if m.DC_offset is not None else 1,
            m.DC_offset if m.DC_offset is not None else float('inf')
        ))

    # Generate colormap and normalizer if color_bar is enabled
    cmap, norm, min_val, max_val = generate_colormaps_and_normalizers(data_in, c_bar)

    # --- Create Figure and Axes using constrained_layout ---
    double_fig_size = (fig_size[0], fig_size[1]/2) # Adjusted for two subplots
    fig, ax = plt.subplots(1, 2, figsize=double_fig_size, constrained_layout=True)
    # If only one axis is returned (e.g., if user passed bad fig_size implicitly)
    if not isinstance(ax, np.ndarray) or ax.size != 2:
         raise ValueError("Failed to create 2 subplots. Check fig_size.")

    # Titles and labels (your existing dict is fine)
    titles = {
        'Zabsphi': (r"$|Z|\,(\Omega)$", r"Phase ($^{\circ}$)"),
        'Zrealimag': (r"$Z'\,(\Omega)$", r"$-Z''\,(\Omega)$"),
        'permittivity': (r"$\varepsilon'$", r"$\varepsilon''$"),
        'tandelta': (r"$\sigma$ (S/m)", r"$\tan\delta$"), # Check if sigma is correct for left axis here
        'modulus': (r"$M'$", r"$M''$")
    }

    if d_type not in titles:
        print(f"Error: Invalid data type '{d_type}'. Valid types are: {list(titles.keys())}")
        plt.close(fig) # Close the empty figure
        return None, None

    ylabels = titles[d_type]

    # Default color cycle if no colorbar
    if not c_bar:
        # Use the default prop_cycle defined in your style
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']


    # --- Plotting Loop ---
    plotted_lines_left = [] # For legend handles
    plotted_lines_right = []

    for i, measurement in enumerate(data_in):
        plot_string = measurement.plot_string  # Label for legend

        # --- Data Extraction (same as before) ---
        if d_type == 'Zabsphi':
            data = measurement.Zabsphi
            x, y1, y2 = data[:, 0], data[:, 1], data[:, 2]
            ax[0].set_yscale('log') # Apply scale here
        elif d_type == 'Zrealimag':
            data = measurement.Zrealimag
            x, y1, y2 = data[:, 0], data[:, 1], -data[:, 2]
            # ax[0].set_yscale('log') # Apply scale if needed
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
        else: # Should not happen due to check above, but good practice
             continue

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

        # Skip plotting if no data remains after masking
        if x1_masked.size == 0 and x2_masked.size == 0:
            print(f"Warning: No data points left for {plot_string} after applying limits.")
            continue

        # --- Determine Color ---
        if c_bar == 1 and measurement.Temperature is not None:
            color = cmap(norm(measurement.Temperature))
        elif c_bar == 2 and measurement.DC_offset is not None:
            color = cmap(norm(measurement.DC_offset))
        else:
            # Use default color cycle if no colorbar or value is None
            color = colors[i % len(colors)] # Cycle through default colors

        # --- Plot ---
        # Plot only if data exists for the specific axis
        line1 = None
        if x1_masked.size > 0:
            line1 = ax[0].semilogx(x1_masked, y1_masked, label=plot_string, color=color)
            plotted_lines_left.extend(line1) # Store handle for legend

        line2 = None
        if x2_masked.size > 0:
            line2 = ax[1].semilogx(x2_masked, y2_masked, label=plot_string, color=color)
            plotted_lines_right.extend(line2) # Store handle for legend


    # --- Axis Labels and Limits ---
    ax[0].set_xlabel('Frequency (Hz)')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[0].set_ylabel(ylabels[0])
    ax[1].set_ylabel(ylabels[1])

    # Apply limits AFTER plotting and AFTER setting scale
    if x_lim:
        ax[0].set_xlim(x_lim)
        ax[1].set_xlim(x_lim)
    if y_lim_left:
        ax[0].set_ylim(y_lim_left)
    if y_lim_right:
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
                 cbar.set_ticklabels([f'{min_val:.1f} K'])
                 cbar.set_label("Temperature (K)")
             else:
                 cbar.set_ticklabels([f'{min_val:.1f} V'])
                 cbar.set_label("DC Offset (V)")
        else:
             if c_bar == 1:
                 cbar.set_ticklabels([f'{min_val:.1f}', f'{max_val:.1f}']) # Just numbers, label indicates units
                 cbar.set_label("Temperature (K)")
             else:
                 cbar.set_ticklabels([f'{min_val:.1f}', f'{max_val:.1f}']) # Just numbers, label indicates units
                 cbar.set_label("DC Offset (V)")

        cbar.minorticks_off()
        cbar.outline.set_linewidth(0.5)
    # --- Legends ---
    # Create legends using the handles collected during plotting
    # This ensures only plotted lines appear in the legend
    if plotted_lines_left:
         # You might want fewer labels in the legend if there are many lines
         # Consider showing only unique labels or a subset
         handles, labels = ax[0].get_legend_handles_labels()
         ax[0].legend(handles, labels) # Or customize further: loc='best', fontsize='small', etc.

    if plotted_lines_right:
         handles, labels = ax[1].get_legend_handles_labels()
         ax[1].legend(handles, labels)

    # Layout and show (constrained_layout handles adjustments before showing)
    plt.show()

    return fig, ax




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
