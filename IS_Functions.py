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

def add_colorbar(fig, cax, cmap, norm, min_val, max_val, c_bar, fig_size):
    """
    Add a colorbar to the figure for both axes.
    """
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_ticks([min_val, max_val])
    
    if c_bar == 1:
        cbar.set_label("Temperature")
        cbar.set_ticklabels([f'{min_val:.1f} K', f'{max_val:.1f} K']) # Temperature
    else:
        cbar.set_label("DC Offset")
        cbar.set_ticklabels([f'{min_val:.1f} V', f'{max_val:.1f} V']) # DC offset
        
    cbar.minorticks_off()  # Remove minor ticks
    cbar.outline.set_linewidth(0.5)

    # Adjust colorbar position and size based on figure size
    if fig_size == [3.5, 2.625]:
        height_scale = 0.8
        pos = cax.get_position()
        new_height = pos.height * height_scale
        new_y0 = pos.y0 + (pos.height - new_height) / 2  # Center the colorbar vertically
        cax.set_position([
            pos.x0 + 0.03,  # Adjust x0 to move the colorbar to the right
            new_y0,  # Center the colorbar vertically
            pos.width * 0.3,  # Adjust width to shrink the colorbar
            new_height  # Adjust height to shrink the colorbar
        ])
    else:
        height_scale = 0.8
        pos = cax.get_position()
        new_height = pos.height * height_scale
        new_y0 = pos.y0 + (pos.height - new_height) / 2  # Center the colorbar vertically
        cax.set_position([
            pos.x0 + 0.03,  # Adjust x0 to move the colorbar to the right
            new_y0,  # Center the colorbar vertically
            pos.width * 0.3,  # Adjust width to shrink the colorbar
            new_height  # Adjust height to shrink the colorbar
        ])

        
def IS_plot(
    data_in: list, 
    d_type: str,
    x_lim: tuple = None,
    y_lim_left: tuple = None, #limit for the left plot y axis
    y_lim_right: tuple = None, #limit for the right plot y axis
    sort_data: bool = True, # Order the data by temperature and DC offset
    fig_size: tuple = (10, 5), # Size of the figure - needed to pad plots correctly for colorbar
    c_bar: float = 0, # 0 = no colorbar, 1 = colorbar for temperature, 2 = colorbar for DC_offset
    
    ):
    '''Plotting function for the impedance data
    data_in: list - the list of Measurement class data to plot
    d_type: str - the type of data to plot: 'Zabsphi', 'Zrealimag', 'permittivity', 'tandelta', 'modulus'
    '''
    
    if not data_in:
        print("No data to plot")
        return

    if sort_data:
        # Sort data first by temperature, then by DC offset, with None values last
        # As it is heirarchical it will sub sort temperature data into DC or just sort by DC if no temperature present
        data_in.sort(key=lambda m: (
        # First sort criterion: data with temperature vs without (None values last)
        0 if m.Temperature is not None else 1,
        # Second: round temperature to nearest 10 (only applies to non-None values)
        round(m.Temperature, -1) if m.Temperature is not None else float('inf'),
        # Third: data with DC offset vs without (None values last in their group)
        0 if m.DC_offset is not None else 1,
        # Fourth: actual DC offset value (only applies to non-None values)
        m.DC_offset if m.DC_offset is not None else float('inf')
        ))
        
   

    # Generate colormap and normalizer if color_bar is enabled
    cmap, norm, min_val, max_val = generate_colormaps_and_normalizers(data_in, c_bar)
    
    # Create the figure and axes including correct scaling for colorbar
    if c_bar == 0:
        # Create the figure and axes using gridspec
        fig = plt.figure(figsize = (fig_size[0], fig_size[1]/2))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])  # Define grid layout
        ax = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    
    # If we are adding a colorbar we add some padding to the right of the second plot so the scaling doesn't change when a single colorbar is added
    else:
        # Keep the first two columns at ratio=1 each, then a slim column for colorbar
        width_ratios = [1, 1, 1/9]
        new_width = fig_size[0] * (sum(width_ratios) / 2.0)  # e.g., if the two columns = 2.0 in ratio
        fig = plt.figure(figsize = (new_width , fig_size[1]/2))
        # add a third subplot for the colorbar
        gs = gridspec.GridSpec(1, 3, width_ratios = width_ratios)
        ax = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]


    # Titles and labels
    titles = {
    'Zabsphi': (r"$|Z|\,(\Omega)$", r"Phase ($^{\circ}$)"),
    'Zrealimag': (r"$Z'\,(\Omega)$", r"$-Z''\,(\Omega)$"),
    'permittivity': (r"$\varepsilon'$", r"$\varepsilon''$"),
    'tandelta': (r"$\sigma$ (S/m)", r"$\tan\delta$"),
    'modulus': (r"$M'$", r"$M''$")
    }
    
    # Check if d_type is valid
    if d_type not in titles:
        print(f"Error: Invalid data type '{d_type}'. Valid types are: {list(titles.keys())}")
        return

    # Assign dict values to labels
    ylabels = titles[d_type]
    
    # Create a colormap from plasma the same length as data_in
    cmap_dat = plt.get_cmap('plasma')(np.linspace(0, 1, len(data_in)))
    
    
    
    for i, measurement in enumerate(data_in, start=0):
        plot_string = measurement.plot_string  # Label for legend

        if d_type == 'Zabsphi':
            data = measurement.Zabsphi
            x, y1, y2 = data[:, 0], data[:, 1], data[:, 2]  # (frequency, |Z|, phase)
            ax[0].set_yscale('log')
        elif d_type == 'Zrealimag':
            data = measurement.Zrealimag
            x, y1, y2 = data[:, 0], data[:, 1], -data[:, 2]  # (frequency, Zreal, Zimag)
            # ax[0].set_yscale('log')
        elif d_type == 'permittivity':
            data = measurement.permittivity
            x, y1, y2 = data[:, 0], data[:, 1], data[:, 2]  # (frequency, ε′, ε″)
        elif d_type == 'tandelta':
            cond_data = measurement.conductivity
            tan_delta_data = measurement.tandelta
            x, y1, y2 = cond_data[:, 0], cond_data[:, 1], tan_delta_data[:, 1]  # (frequency, σ, tan δ)
        elif d_type == 'modulus':
            data = measurement.modulus
            x, y1, y2 = data[:, 0], data[:, 1], data[:, 2]  # (frequency, M′, M″)
        else:
            print('Invalid data type')
            return

        # Mask the data based on the x_limits if given
        if x_lim:
            x_mask = (x >= x_lim[0]) & (x <= x_lim[1])
            x, y1, y2 = x[x_mask], y1[x_mask], y2[x_mask]
        # Separate x-axes for left and right y-axes
        x1, x2 = x.copy(), x.copy()
        # Mask the data based on the y_limits if given
        if y_lim_left:
            y1_mask = (y1 >= y_lim_left[0]) & (y1 <= y_lim_left[1])
            x1, y1 = x1[y1_mask], y1[y1_mask]
        if y_lim_right:
            y2_mask = (y2 >= y_lim_right[0]) & (y2 <= y_lim_right[1])
            x2, y2 = x2[y2_mask], y2[y2_mask]

        # Plot
        color = cmap(norm(measurement.Temperature if c_bar == 1 else measurement.DC_offset)) if c_bar else cmap_dat[i]
        ax[0].semilogx(x1, y1, label=plot_string, color=color)
        ax[1].semilogx(x2, y2, label=plot_string, color=color)

    # Set axis labels
    ax[0].set_xlabel('Frequency (Hz)')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[0].set_ylabel(ylabels[0])
    ax[1].set_ylabel(ylabels[1])
    


    # Add colorbar if enabled
    if c_bar:
        add_colorbar(fig, ax[2], cmap, norm, min_val, max_val, c_bar, fig_size)

    # Legends
    ax[0].legend()#loc='best', fontsize='small', markerscale=0.8, framealpha=0.4)
    ax[1].legend()#loc='best', fontsize='small', markerscale=0.8, framealpha=0.4)

    # Layout and show
    plt.show()
    return fig, ax


