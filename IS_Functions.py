#import all the libraries needed
from import_dep import *


        
def IS_plot(data_in: list, d_type: str = 'Zabsphi'):
    '''Plotting function for the impedance data
    data_in: list - the list of Measurement class data to plot
    d_type: str - the type of data to plot: 'Zabsphi', 'Zrealimag', 'permittivity', 'conductivity', 'modulus'
    v_type: str - the thing that is varied and being compared: 'temperature', 'DC_offset', 'state', 'run_number'
    index: int - the index of the data to plot
    '''
    

    # Hierarchically sort the list of measurements first by rounded temperature, then by DC offset (note the rounding is to the nearest 10 due to accuracy limitations)
    data_in.sort(key=lambda m: (round(m.Temperature, -1), m.DC_offset))

    # Create the figure and axis
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Iterate over the data and append the data to the plots
    
    for i, measurement in enumerate(data_in):
        
        # Extract the data for the given index
        if d_type == 'Zabsphi':
            Zabsphi = measurement.Zabsphi
            plot_string = measurement.plot_string
            ax[0].plot(Zabsphi[:, 2], Zabsphi[:, 4], label=plot_string)
            ax[1].plot(Zabsphi[:, 2], Zabsphi[:, 5], label=plot_string)
        elif d_type == 'Zrealimag':
            Zrealimag = measurement.Zrealimag
            plot_string = measurement.plot_string
            ax[0].plot(Zrealimag[:, 2], Zrealimag[:, 1], label=plot_string)
            ax[1].plot(Zrealimag[:, 2], Zrealimag[:, 2], label=plot_string)
        elif d_type == 'permittivity':
            permittivity = measurement.permittivity
            plot_string = measurement.plot_string
            ax[0].plot(permittivity[:, 2], permittivity[:, 1], label=plot_string)
            ax[1].plot(permittivity[:, 2], permittivity[:, 2], label=plot_string)
        elif d_type == 'conductivity':
            conductivity = measurement.conductivity
            plot_string = measurement.plot_string
            ax[0].plot(conductivity[:, 2], conductivity[:, 1], label=plot_string)
            ax[1].plot(conductivity[:, 2], conductivity[:, 2], label=plot_string)
        elif d_type == 'modulus':
            modulus = measurement.modulus
            plot_string = measurement.plot_string
            ax[0].plot(modulus[:, 2], modulus[:, 1], label=plot_string)
            ax[1].plot(modulus[:, 2], modulus[:, 2], label=plot_string)
        else:
            print('Invalid data type')
            return
    # Extract the data for the given index
    Zabsphi = self.Zabsphi[index]
    Zrealimag = self.Zrealimag[index]
    run_number = self.run_number[index]
    DC_offset = self.DC_offset[index]
    Temperature = self.Temperature[index]
    plot_string = self.plot_string[index]
    
    # Create the figure and axis
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot the Zabs data
    ax[0].plot(Zabsphi[:, 2], Zabsphi[:, 4], label='Zabs')
    ax[0].set_xlabel('Frequency (Hz)')
    ax[0].set_ylabel('Zabs (Ohms)')
    ax[0].set_title(f'Zabs vs Frequency {plot_string}')
    ax[0].legend()
    
    # Plot the phi data
    ax[1].plot(Zabsphi[:, 2], Zabsphi[:, 5], label='phi')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('phi (degrees)')
    ax[1].set_title(f'phi vs Frequency {plot_string}')
    ax[1].legend()
    
    plt.tight_layout()
    plt.show()
        
