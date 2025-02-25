#import all the libraries needed
from import_dep import *


class IS_Agilent:
    '''Class to store, process and plot XRD RSM data
    '''
    def __init__(self, root_IS_Agilent: str, folder_name: str):
        # Initialize the class with the root folder and the data folder
        self.root_IS: Path = Path(root_IS_Agilent)
        self.folder_name: Path = Path(folder_name)
        # Construct the full path to the folder from the root directory and folder name
        self.folder_path: Path = self.root_IS / self.folder_name
           
        # Tuple storing tuples of IS data, in various forms, for each file within the selected folder
        self.Zabstheta: tuple = None
        self.Zrealimag: tuple = None
        self.Epsilon: tuple = None
        self.Modulus: tuple = None
        
        # Tuple storing the data frames of the imported data for debugging
        self.Zabstheta_df: tuple = None
        
        # tuple to store the attributes of the IS measurement such as the DC offset and Temperature
        self.run_number: tuple = None
        self.DC_offset: tuple = None 
        self.Temperature: tuple = None 
        self.plot_string: tuple = None
        
    
          
        # Load the data as you initialize the class
        self._load_data()
        #self._transform_data()


    def _load_data(self) -> None:
        '''Load data from all the csv/txt files in the folder and store it in the class as Zabs and theta data
        From the file name various attribues such as temperature as stored along with it
        '''
        
        Zabstheta_df = [] # list to store the Zabs and theta data data frame for debugging
        Zabstheta = [] # list to store the Zabs and theta data
        run_number = [] # list to store the run number
        DC_offset = [] # list to store the DC offset
        Temperature = [] # list to store the temperature
        plot_string = [] # list to store the plot string
        
        # Get a list of all the files in the folder as an iterator of path objects
        files = [f for f in self.folder_path.iterdir() if f.suffix in ['.csv']]
        # Check if no files were found
        if not files:
            print("Error: No files with .csv extension were found.")
            return
    
        # Sort the files alphabetically
        files.sort()
        
        print(files)
        
        for i, fi in enumerate(files):
            # Load the data from the file
            
            try:
                # Read the data starting from the header line
                df = pd.read_csv(fi, sep=',', skiprows=0, header=None)    
                # Drop columns that contain only NaN values
                df.dropna(axis=1, how='all', inplace=True)
                # Rename the columns
                df.columns = ['timer','NA','frequency','NA2','Zabs', 'theta']
                
            except Exception as e:
                print(f"Error with file: {fi}, {e}")
                continue
            
            print(df.head(3))
            
            # Append the data to the lists
            Zabstheta.append(df.to_numpy())
            Zabstheta_df.append(df)
            
            
            # Extract the run number, DC offset and Temperature from the file name
            run_number_match = re.search(r'run(\d+)', fi.stem)
            DC_offset_match = re.search(r'bias_([\d.]+)', fi.stem)
            Temperature_match = re.search(r'temp_([\d.]+)', fi.stem)
            
            # Use default values if not found
            run_number_val = float(run_number_match.group(1)) if run_number_match else 0.0 
            DC_offset_val = float(DC_offset_match.group(1)) if DC_offset_match else 0.0
            Temperature_val = float(Temperature_match.group(1)) if Temperature_match else 0.0
            
            # Round the extracted values to 1 decimal place
            run_number_val = round(run_number_val, 2)
            DC_offset_val = round(DC_offset_val, 2)
            Temperature_val = round(Temperature_val, 2)
            
            # Append the rounded values to the lists
            run_number.append(run_number_val)
            DC_offset.append(DC_offset_val)
            Temperature.append(Temperature_val)
            # Generate the plot string using the extracted values then append to list
            plot_string.append(f'run={run_number_val}, DC={DC_offset_val}, T={Temperature_val}')
            
            print(plot_string[-1])
            
            
            
        
        self.Zabstheta = Zabstheta
        self.Zabstheta_df = Zabstheta_df

        self.file_name = [str(p) for p in files] # store the file names as strings not path objects
        self.plot_string = [p.stem for p in files ]
        
        self.run_number = run_number
        self.DC_offset = DC_offset
        self.Temperature = Temperature
        self.plot_string = plot_string
        
    # Functions to convert between reciprocal space coordinates and lattice parameters
    @staticmethod
    def Qx_to_a(Qx):
        return (2 * np.pi) / Qx  # a = 2π/Qx
    @staticmethod
    def a_to_Qx(a):
        return (2 * np.pi) / a
    @staticmethod
    def Qz_to_c(Qz):
        return (6 * np.pi) / Qz  # c = 6π/Qz (from l=3)
    @staticmethod
    def c_to_Qz(c):
        return (6 * np.pi) / c
    #####

    # def _transform_data(self):
    #     if not self.RSM_df:
    #         print("No data to extract")
    #         return
        
    #     lat_param_df = []
    #     lat_param_np = []
    #     q_df = []
    #     q_np = []
        
    #     for d in self.RSM_df:
                          
    #         # Step 1: Extract the values from the dataframe
    #         two_theta = np.radians(d['2Theta position'].values)  # Convert to radians
    #         omega = np.radians(d['Omega position'].values)  # Convert to radians
    #         intensity = d['Intensity'].values

    #         # Step 2: Calculate q_x and q_z for reciprocal space coordinates
    #         q_x = ((4 * np.pi) / self.wavelength) * (np.sin(two_theta / 2) * np.sin((two_theta / 2) - omega))
    #         q_z = ((4 * np.pi) / self.wavelength) * (np.sin(two_theta / 2) * np.cos((two_theta / 2) - omega))
    #         #q_x = ((2 * np.pi) / self.wavelength) * (np.cos(omega) - np.cos(two_theta-omega)) # For the other convention
    #         #q_z = ((2 * np.pi) / self.wavelength) * (np.sin(omega) + np.sin(two_theta-omega)) # For the other convention
            
    #         # Step 3: Calculate a and c lattice parameters from q_x and q_z based off this being the (103 peak)
    #         a = self.Qx_to_a(q_x)
    #         c = self.Qz_to_c(q_z)
            
    #         # Step 4: Store the q data in a dataframe and append to the lists
    #         df_q_params = pd.DataFrame({'qx': q_x, 'qz': q_z, 'Intensity': intensity})
    #         q_df.append(df_q_params)
    #         q_np.append(df_q_params.to_numpy())
            
    #         # Step 5: Store the lattice parameters in a dataframe and append to the lists
    #         df_lat = pd.DataFrame({'a': a, 'c': c, 'Intensity': intensity})
    #         lat_param_df.append(df_lat)
    #         lat_param_np.append(df_lat.to_numpy())
            
            
    #     self.qxqz_df = q_df
    #     self.qxqz_np = q_np
    #     self.lat_param_df = lat_param_df
    #     self.lat_param_np = lat_param_np      
