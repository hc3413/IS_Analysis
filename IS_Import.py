#import all the libraries needed
from import_dep import *


@dataclass
class ISdata:
    """Dataclass to store a single impedance measurement
    Regardless of the instrument used, the data is stored in the same format and will have 
    all of the properties below"""
    # Non-optional attributes
    file_name: str
    run_number: int
    Zabsphi: np.ndarray  # (frequency, Zabs, phi)
    folder_path: Path # Path to the data, where we will also export the data
    
    # Optional device metadata attributes
    DC_offset: float = None
    V_rms: float = None # AC voltage applied
    Temperature: float = None
    res_state: str = None
    vac_state: str = None
    device_name: str = None
    amp_state: str = None
    
    # Calculated transformed data attributes
    Zcomplex: np.ndarray = None  # (frequency, Zreal + j*Zimag), computed later
    Zrealimag: np.ndarray = None  # (frequency, Zreal, Zimag), computed later
    permittivity: np.ndarray = None  # (frequency, Real_permittivity, Imag_permittivity), computed later
    tandelta: np.ndarray = None  # (frequency, tan delta), computed later
    conductivity: np.ndarray = None  # (frequency, conductivity, loss), computed later
    modulus: np.ndarray = None  # (frequency, modulus, phase), computed later
    
    # Fitted data attributes
    Zcomplex_fit: np.ndarray = None  # (frequency, Zcomplex), fitted by a model
    Z_parameters: dict = None  # Fitted parameters from the model (R_mem, C_mem, R_mem2, C_mem2, R_series, C_pad, Q1, Q2, alpha1, alpha2 )
    
    # Tuple storing the data frames of the imported data for debugging
    Zabsphi_df: pd.DataFrame = None
    
    _plot_string_override: str = ""  # store user override for plot_string
    
    # Generate a plot label string based on metadata
    @property
    def plot_string(self) -> str:
        """Generate a plot label string based on metadata, skipping None values."""
        
        if self._plot_string_override: # if the user has set a plot string override, use that
            return self._plot_string_override
        
        # Otherwise, generate the plot string from the metadata
        parts = []
        if self.run_number is not None:
            parts.append(f"run={self.run_number}")
        if self.DC_offset is not None:
            parts.append(f"DC={self.DC_offset}")
        if self.V_rms is not None:
            parts.append(f"V_rms={self.V_rms}")
        if self.Temperature is not None:
            parts.append(f"T={self.Temperature}")
        if self.res_state is not None:
            parts.append(self.res_state)
        if self.vac_state is not None:
            parts.append(self.vac_state)
        if self.amp_state is not None:
            parts.append(self.amp_state)
        if self.device_name is not None:
            parts.append(self.device_name)
        return ", ".join(parts)
    
    @plot_string.setter
    def plot_string(self, value: str):
        self._plot_string_override = value if value else self._plot_string_override

class ImpedanceData(ABC):
    """Abstract base class for storing and processing impedance data from all instruments."""

    def __init__(self):
        self.measurements: dict[int, ISdata] = {}  # Store data in a dictionary indexed by a number from 0, 1, 2, ...

    def __iter__(self):
        """Allow direct iteration over measurement objects.
        you can directly call for m in imported_data_object and it will return the ISdata objects:"""
        return iter(self.measurements.values())

    def __getitem__(self, index_number: int) -> ISdata:
        """Safely retrieve a measurement from its Index number, with error handling.
        you can directly call imported_data_object[0] to get the first measurement ISdata object"""
        if index_number not in self.measurements:
            raise KeyError(f"Index number {index_number} not found.")
        return self.measurements[index_number]

    @abstractmethod
    def _load_data(self):
        """Abstract method that must be implemented by subclasses to load data."""
        pass


    def _transform_data(self):
        """Transform (Zabs, phi) to: (Zreal, Zimag), permittivity, tandelta, conductivity, modulus
        Then update the measurement objects with this data."""
        if not self.measurements:
            print("No data to transform")
            return

        for measurement in self:
            # Constants
            eps0 = 8.854e-12  # Vacuum permittivity (F/m)
            Cap_0 = eps0*((20e-6)**2)/(30e-9)  # Vacuum capacitance 

            # Compute Zreal, Zimag from Zabs, phi
            Zap = np.copy(measurement.Zabsphi) # Copy the data to avoid modifying the original            
            Z_complex = Zap[:, 1]*np.exp(1j*np.radians(Zap[:, 2])) # Combine Zreal and Zimag to form complex impedance Z
            measurement.Zcomplex = np.column_stack((Zap[:, 0], Z_complex))  # (frequency, Zreal + j*Zimag)
            measurement.Zrealimag = np.column_stack((Zap[:, 0], np.real(Z_complex), np.imag(Z_complex)))  # (frequency, Zreal, Zimag)

            # Compute permittivity (real and imaginary parts)
            omega = 2 * np.pi * Zap[:, 0]
            epsilon = 1/(1j*omega*Cap_0 * Z_complex)
            epsilon_real = np.real(epsilon)  # 
            epsilon_imag = -np.imag(epsilon)  # e = e' - je''
            measurement.permittivity = np.column_stack((Zap[:, 0], epsilon_real, epsilon_imag))

            # Compute tan delta
            tandelta = epsilon_imag / (epsilon_real + 1e-20)  # Prevent division by zero
            measurement.tandelta = np.column_stack((Zap[:, 0], tandelta))

            # Compute conductivity
            conductivity = omega * eps0 * epsilon_imag
            measurement.conductivity = np.column_stack((Zap[:, 0], conductivity))

            # Compute electric modulus (real and imaginary parts)
            Modulus_complex = 1/epsilon
            measurement.modulus = np.column_stack((Zap[:, 0], np.real(Modulus_complex), np.imag(Modulus_complex)))
    

    def plot(self, index_numbers: tuple):
        """Plot Zabs and phi vs frequency for a tuple of given run numbers on a log-log scale.
        This is just a simple plotting function for rough plots
        more advanced plotting is done with the IS_plot function in IS_Functions.py"""
        

        cmap_dat = plt.get_cmap('plasma')(np.linspace(0, 1, len(index_numbers)))
        
        # Create a figure with two subplots
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # Iterate over the index numbers and append the data to the plots
        for i, idx in enumerate(index_numbers):
            
            if i not in self.measurements:
                print(f"No data found for index number {i}")
                return
            
            # Extract the measurement class object from the dict for the given index
            measurement = self.measurements[idx]
            Zabsphi = measurement.Zabsphi

            # Plot Zabs vs frequency
            ax[0].loglog(Zabsphi[:, 0], Zabsphi[:, 1], color=cmap_dat[i], label=r'$|Z|$ - ' f' {measurement.plot_string}')
            # Plot phi vs frequency
            ax[1].semilogx(Zabsphi[:, 0], Zabsphi[:, 2], color=cmap_dat[i], label=r'$\phi$ - ' f' {measurement.plot_string}')
        
        
        ax[0].set_xlabel('Frequency (Hz)')
        ax[0].set_ylabel('Zabs (Ohms)')
        ax[0].set_title(f'Zabs vs Frequency {measurement.plot_string}')
        #ax[0].legend()
        
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].set_ylabel('phi (degrees)')
        ax[1].set_title(f'phi vs Frequency {measurement.plot_string}')
        #ax[1].legend()

        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def _extract_value(filename: str, pattern: str, default: float) -> float:
        """Helper method to extract numeric values from filenames."""
        match = re.search(pattern, filename)
        # Return the extracted value if found rounded to 2dp, otherwise return the default value
        return round(float(match.group(1)), 2) if match else default

    @staticmethod
    def _extract_string(filename: str, patterns: tuple, default: str) -> str:
        """Helper method to extract string values from filenames.
        Searches for a match within a tuple of patterns."""
        filename_lower = filename.lower()
        # Return the first matching pattern or the default value if none found
        return next((p for p in patterns if p.lower() in filename_lower), default)  
    
    


class AgilentIS(ImpedanceData):
    """Class for importing and processing Agilent impedance spectroscopy data."""

    def __init__(self, root_folder: str, folder_name: str):
        super().__init__()
        self.folder_path = Path(root_folder) / Path(folder_name)
        self._load_data()
        self._transform_data()

    def _load_data(self):
        """Load data from all CSV files in the folder."""
        # Get a list of all the files in the folder as an iterator of path objects and sort alphabetically
        files = sorted([f for f in self.folder_path.iterdir() if f.suffix == '.csv'])
        
        # Print error if no files were found
        if not files:
            print("Error: No CSV files found.")
            return
        

        for idx, fi in enumerate(files, start=0):
            try:
                # Read the data starting from the header line
                df = pd.read_csv(fi, sep=',', skiprows=0, header=None)
                # Drop columns that contain only NaN values
                df.dropna(axis=1, how='all', inplace=True)
                # Rename the columns with frequency input in degrees
                df.columns = ['timer', 'NA', 'frequency', 'NA2', 'Zabs', 'phi']
                
                #print(df.head(3))
                
                # Extract the run number from the filename
                run_number = self._extract_value(fi.stem, r'run(\d+)', default=0)
                # Create a ISdata object using the extracted data
                measurement = ISdata(
                    file_name=str(fi),
                    run_number=int(run_number),
                    Zabsphi=df[['frequency', 'Zabs', 'phi']].to_numpy(),
                    folder_path = self.folder_path,
                    DC_offset=self._extract_value(fi.stem, r'bias_([\d.]+)', default=0),
                    Temperature=self._extract_value(fi.stem, r'temp_([\d.]+)', default=0),
                    Zabsphi_df = df  # Store the data frame for debugging
                )
                
                # Store the ISdata object in the dictionary indexed by an integer
                self.measurements[idx] = measurement 
                
                print(f"run={measurement.run_number}, DC={measurement.DC_offset}, T={measurement.Temperature}'")

            except Exception as e:
                print(f"Error loading file {fi}: {e}")
                continue

    

class SolatronIS(ImpedanceData):
    """Class for importing and processing Solatron impedance spectroscopy data."""

    def __init__(self, root_folder: str, folder_name: str):
        super().__init__()
        self.folder_path = Path(root_folder) / Path(folder_name)
        self._load_data()
        self._transform_data()

    def _load_data(self):
        """Load data from all CSV files in the folder."""
        # Get a list of all the files in the folder as an iterator of path objects and sort alphabetically
        files = sorted([f for f in self.folder_path.iterdir() if f.suffix == '.csv'])
        
        # Print error if no files were found
        if not files:
            print("Error: No CSV files found.")
            return
        
        # Initiate a counter
        counter = 0
        for fi in files:
            try:
                # Read the data starting from the header line
                df = pd.read_csv(fi, sep=',', skiprows=4, header=None)
                df = df.iloc[:, :14]  # Keep only the first 14 columns
                # Drop columns that contain only NaN values
                #df.dropna(axis=1, how='all', inplace=True)
                # Rename the columns with frequency input in degrees
                df.columns = [
                    "Result Number", 
                    "Sweep Number", 
                    "Point Number", 
                    "Time", 
                    "frequency", 
                    "AC Level (V)", 
                    "DC Level (V)", 
                    "Set Point", 
                    "Temperature", 
                    "Control", 
                    "Zabs", 
                    "phi", 
                    "Admittance Magnitude (S)", 
                    "Capacitance Magnitude (F)"
                    ]
                # Extract the run number from the filename
                run_number = self._extract_value(fi.stem, r'run(\d+)', default=0)
                # Extract the total number of sweep numbers
                total_sweeps = df['Sweep Number'].nunique()
                # Extract the total number of points per sweep
                total_points = df['Point Number'].nunique()
                
                # Loop over the sweep numbers storing a new ISdata object for each sweep
                for sweep_no in range(1,total_sweeps+1):
                    # Filter the data frame for the current sweep number
                    df_sweep = df[df['Sweep Number'] == sweep_no]
                    # Check that the total number of points is the same as the number we expect - if not then skips the sweep
                    if len(df_sweep) != total_points:
                        print(f"Error: file{fi.name}, sweep {sweep_no} does not have the expected number of points.")
                        continue
                    
                    # Extract the temperature value if it is not a Nan
                    temperature_value = df_sweep['Temperature'].iloc[0]
                    
                    # Convert the temperature from Kelvin to Celsius if it is not a Nan
                    if temperature_value == '-':
                        temperature_kelvin = None
                    else:
                        temperature_kelvin = float(temperature_value) - 273.15

                    # Create a ISdata object using the extracted data
                    measurement = ISdata(
                        file_name =str(fi.name),
                        run_number =int(run_number),
                        Zabsphi= df_sweep[['frequency', 'Zabs', 'phi']].to_numpy(),
                        folder_path = self.folder_path,
                        DC_offset = df_sweep['DC Level (V)'].iloc[0], # Extract the DC level from the first row of the sweep
                        V_rms = df_sweep['AC Level (V)'].iloc[0], # Extract the AC level from the first row of the sweep
                        Temperature = temperature_kelvin,
                        Zabsphi_df = df_sweep,  # Store the data frame for debugging
                        res_state = self._extract_string(fi.stem, ('pristine, electroformed, doubleformed, formed'), default=None), #finds first match for state in tuple
                        vac_state = self._extract_string(fi.stem, ('vacuum', 'ambient', 'vac'), default=None), #finds first match for state in tuple
                        amp_state = self._extract_string(fi.stem, ('noamp', 'amp'), default=None), #finds first match for state in tuple
                        device_name = self._extract_string(fi.stem, ('wirebond1','wirebond2','wirebond3', 'wirebond4','wirebond5','wirebond6', 'wirebond1v2','wirebond2v2', 'wirebond3v2','wirebond4v2'), default=None), #finds first match for state in tuple
                    )
                    
                    # Store the ISdata object in the dictionary indexed by an integer
                    self.measurements[counter] = measurement 
                    counter += 1 #increment the counter for the next measurement
                    
                

            except Exception as e:
                print(f"Error loading file {fi.name}: {e}")
                continue