#import all the libraries needed
from import_dep import *


@dataclass
class Measurement:
    """Dataclass to store a single impedance measurement
    Regardless of the instrument used, the data is stored in the same format and will have 
    all of the properties below"""
    file_name: str
    run_number: int
    DC_offset: float
    Temperature: float
    Zabsphi: np.ndarray  # (frequency, Zabs, phi)
    Zrealimag: np.ndarray = None  # (frequency, Zreal, Zimag), computed later
    permittivity: np.ndarray = None  # (frequency, Real_permittivity, Imag_permittivity), computed later
    tandelta: np.ndarray = None  # (frequency, tan delta), computed later
    conductivity: np.ndarray = None  # (frequency, conductivity, loss), computed later
    modulus: np.ndarray = None  # (frequency, modulus, phase), computed later
    
    # Tuple storing the data frames of the imported data for debugging
    Zabsphi_df: pd.DataFrame = None

    @property
    def plot_string(self) -> str:
        """Generate a plot label string based on metadata."""
        return f'run={self.run_number}, DC={self.DC_offset}, T={self.Temperature}'


class ImpedanceData(ABC):
    """Abstract base class for storing and processing impedance data from all instruments."""

    def __init__(self):
        self.measurements: dict[int, Measurement] = {}  # Store data in a dictionary indexed by a number from 0, 1, 2, ...

    def __iter__(self):
        """Allow direct iteration over measurement objects."""
        return iter(self.measurements.values())

    def __getitem__(self, index_number: int) -> Measurement:
        """Safely retrieve a measurement from its Index number, with error handling."""
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
            # Compute Zreal, Zimag from Zabs, phi
            Zabsphi = measurement.Zabsphi
            Zreal = Zabsphi[:, 1] * np.cos(np.radians(Zabsphi[:, 2]))
            Zimag = Zabsphi[:, 1] * np.sin(np.radians(Zabsphi[:, 2]))
            measurement.Zrealimag = np.column_stack((Zabsphi[:, 0], Zreal, Zimag))  # (frequency, Zreal, Zimag )
            
            # Compute permittivity from Zabs, phi
            permittivity = Zabsphi[:, 1] / (2 * np.pi * Zabsphi[:, 0] * 8.854e-12)
            measurement.permittivity = np.column_stack((Zabsphi[:, 0], permittivity.real, permittivity.imag))  # (frequency, Real_permittivity, Imag_permittivity)
            
            # Compute tan delta from permittivity
            tandelta = measurement.permittivity[:, 2] / measurement.permittivity[:, 1]
            measurement.tandelta = np.column_stack((Zabsphi[:, 0], tandelta))  # (tan delta, frequency)
            
            # Compute conductivity from Zabs, phi
            conductivity = Zabsphi[:, 1] / (2 * np.pi * Zabsphi[:, 0] * 8.854e-12) * 2 * np.pi * Zabsphi[:, 0]
            loss = conductivity / (2 * np.pi * Zabsphi[:, 0] * 8.854e-12)
            measurement.conductivity = np.column_stack((Zabsphi[:, 0], conductivity, loss))  # (frequency, conductivity, loss)
            
            # Compute modulus from permittivity
            modulus = np.sqrt(measurement.permittivity[:, 1] ** 2 + measurement.permittivity[:, 2] ** 2)
            phase = np.degrees(np.arctan(measurement.permittivity[:, 2] / measurement.permittivity[:, 1]))
            measurement.modulus = np.column_stack((Zabsphi[:, 0], modulus, phase))  # (frequency, modulus, phase)
            

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
                # Rename the columns
                df.columns = ['timer', 'NA', 'frequency', 'NA2', 'Zabs', 'phi']
                
                #print(df.head(3))
                
                # Extract the run number from the filename
                run_number = self._extract_value(fi.stem, r'run(\d+)', default=0)
                # Create a Measurement object using the extracted data
                measurement = Measurement(
                    file_name=str(fi),
                    run_number=int(run_number),
                    DC_offset=self._extract_value(fi.stem, r'bias_([\d.]+)', default=0),
                    Temperature=self._extract_value(fi.stem, r'temp_([\d.]+)', default=0),
                    Zabsphi=df[['frequency', 'Zabs', 'phi']].to_numpy(),
                    Zabsphi_df = df  # Store the data frame for debugging
                )
                
                # Store the measurement object in the dictionary indexed by an integer
                self.measurements[idx] = measurement 

            except Exception as e:
                print(f"Error loading file {fi}: {e}")
                continue

    @staticmethod
    def _extract_value(filename: str, pattern: str, default: float) -> float:
        """Helper method to extract numeric values from filenames."""
        match = re.search(pattern, filename)
        # Return the extracted value if found rounded to 2dp, otherwise return the default value
        return round(float(match.group(1)), 2) if match else default


class SolatronIS(ImpedanceData):
    """Class for importing and processing Solatron impedance spectroscopy data."""

    def __init__(self, root_folder: str, file_name: str):
        super().__init__()
        self.file_path = Path(root_folder) / file_name
        self._load_data()
        self._transform_data()

    def _load_data(self):
        """Load data from a single file with multiple runs stored in columns."""
        if not self.file_path.exists():
            print(f"Error: File {self.file_path} does not exist.")
            return

        try:
            df = pd.read_csv(self.file_path, sep=',', header=0)
        except Exception as e:
            print(f"Error loading file {self.file_path}: {e}")
            return

        frequency = df.iloc[:, 0].values  # Frequency is in the first column

        for i in range(1, df.shape[1], 2):  # Assuming alternating columns (Zabs, phi)
            Zabs = df.iloc[:, i].values
            phi = df.iloc[:, i + 1].values if i + 1 < df.shape[1] else np.zeros_like(Zabs)

            run_number = i // 2 + 1
            measurement = Measurement(
                file_name=str(self.file_path),
                run_number=run_number,
                DC_offset=0,  # No DC bias data in this format
                Temperature=0,  # No temperature data in this format
                Zabsphi=np.column_stack((frequency, Zabs, phi)),
                Zabsphi_df = pd.DataFrame({'frequency': frequency, 'Zabs': Zabs, 'phi': phi})  # Store the data frame
            )

            self.measurements[i] = measurement
