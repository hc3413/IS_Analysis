#import all the libraries needed
from import_dep import *


        
def IS_plot(
    data_in: list, 
    d_type: str,
    x_lim: tuple = None,
    y_lim: tuple = None,
    sort_data: bool = True, # Order the data by temperature and DC offset
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
        
    # Create the figure and axes
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

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
    
    for i, measurement in enumerate(data_in, start = 0):
        plot_string = measurement.plot_string  # Label for legend

        if d_type == 'Zabsphi':
            data = measurement.Zabsphi
            x, y1, y2 = data[:, 0], data[:, 1], data[:, 2]  # (frequency, |Z|, phase)
            ax[0].set_yscale('log')
        elif d_type == 'Zrealimag':
            data = measurement.Zrealimag
            x, y1, y2 = data[:, 0], data[:, 1], -data[:, 2]  # (frequency, Zreal, Zimag)
            #ax[0].set_yscale('log')
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
        
        # Mask the data based off the x_limits if given
        if x_lim:
            x_mask = (x >= x_lim[0]) & (x <= x_lim[1])
            x, y1, y2 = x[x_mask], y1[x_mask], y2[x_mask]

        # Plot
        ax[0].semilogx(x, y1, label=plot_string, color=cmap_dat[i])
        ax[1].semilogx(x, y2, label=plot_string, color=cmap_dat[i])

    # Set axis labels
    ax[0].set_xlabel('Frequency (Hz)')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[0].set_ylabel(ylabels[0])
    ax[1].set_ylabel(ylabels[1])


    # Legends
    ax[0].legend(loc='best', fontsize='small', markerscale=0.8, framealpha=0.4)
    ax[1].legend(loc='best', fontsize='small', markerscale=0.8, framealpha=0.4)

    # Layout and show
    plt.tight_layout()
    plt.show()
    return fig, ax