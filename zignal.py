
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Zignal():
    """Class that defines a Zignal (signal) object.
    
    [I'm calling it a Zignal because "signal" is already something in the standard python library LMAO]
    """
    
    def __init__(self, raw_data: pd.DataFrame, 
                 initials: str,
                 date: str, 
                 trial_type: str, 
                 trial_no: int,
                 fold: int, 
                 sampling_frequency: float):
        """Initializes Zignal object with relevant metadata

        Args:
            raw_data (pd.DataFrame): raw dataframe read in from the WITMOTION file
            initials (str): initials of participant (e.g. vv)
            date (str): date of trial (e.g. 120924)
            trial_type (str): type of trial (e.g. normbrush, bkcomb)
            trial_no (int): trial number
            sampling_frequency (float): Fs 
        """
        self.raw_data = raw_data
        self.initials = initials
        self.date = date
        self.trial_type = trial_type
        self.trial_no = trial_no
        self.fs = sampling_frequency
        self.fold = fold 

        #save individual components
        self.xa = self._get_xacc()
        self.ya = self._get_yacc()
        self.za = self._get_zacc()

        self.xg = self._get_xgyro()
        self.yg = self._get_ygyro()
        self.zg = self._get_zgyro()
        

        if "bk" in self.trial_type: #for classification 
            self.level = 1 #if bk/impaired, note as one (yes) --> this is sorta like one hot encoding, ig.
        else:
            self.level = 0
        
        self.enmoed = self.rm_outs(self.enmo(self.mag(self.raw_data))) #signal with enmo applied for later analysis...

        #magnitude of gyro
        self.gyro = self.rm_outs(self.mag_gyro())

        
    def mag(self, df: pd.DataFrame) -> pd.Series:
        """Calculates the magnitude of a Zignal. Automatically called on init.
        Returns:
            np.ndarray: magnitude of signal
        """
        try:
            return(np.sqrt(df["AccX(g)"] **2  + df["AccY(g)"] **2  + df["AccZ(g)"] ** 2))
        except:
            return(np.sqrt(df["Acceleration X(g)"] **2  + df["Acceleration Y(g)"] **2  + df["Acceleration Z(g)"] ** 2))
    
    def mag_gyro(self) -> np.ndarray:
        """ Calculates the magnitude of the gyroscope signal. No need for enmo?

            Returns:
                np.ndarray: magnitude of gyro signal
        """
        mag_gyro = np.sqrt(self.xg ** 2 + self.yg ** 2 +  self.zg ** 2)
        
        return mag_gyro

    def enmo(self, s: pd.Series) -> pd.Series:
        """Returns inputted signal minus one. Used for gravity compensation. 
        Only to be used with mag()

        Args:
            s (np.ndarray): input signal

        Returns:
            np.ndarray: signal - 1
        """
        return (s - 1)
        
        
    def rm_outs(self, s: pd.Series, devs: int = 3) -> pd.Series:
        """Removes outliers from input signal and returns the signal.

        Args:
            s (np.ndarray): Input signal
            devs (int, optional): Number of standard deviations for outlier cutoff. Defaults to 3.

        Returns:
            np.ndarray: Signal with outliers removed.
        """ 
        s_mean = np.mean(s)
        
        s = np.where(s > np.mean(s) - devs * np.std(s), s, s_mean)
        s = np.where(s < np.mean(s) + devs * np.std(s), s, s_mean)

        
        return s
        
    def __str__(self):
        """string representation of each Zignal. Returned when print(Zignal) occurs.
        """
        return (f"{self.trial_type} by {self.initials} on {self.date} trial {self.trial_no}")
    
    #-------
    
    def _get_xacc(self):
        try:
            xa = self.raw_data["AccX(g)"] 
        except: 
            xa = self.raw_data["Acceleration X(g)"]
        return xa
    
    def _get_yacc(self):
        try:
            ya = self.raw_data["AccY(g)"] 
        except: 
            ya = self.raw_data["Acceleration Y(g)"]
        return ya

    def _get_zacc(self):
        try:
            za = self.raw_data["AccZ(g)"] 
        except: 
            za = self.raw_data["Acceleration Z(g)"]
        return za

    #-------

    def _get_xgyro(self): 
        if "AsX(Â°/s)" in self.raw_data.columns.tolist():
            return self.raw_data["AsX(Â°/s)"]
        else:
            return self.raw_data['AsX(°/s)']
        
    def _get_ygyro(self): 
        if "AsY(Â°/s)" in self.raw_data.columns.tolist():
            return self.raw_data["AsY(Â°/s)"]
        else:
            return self.raw_data['AsY(°/s)']
        
    def _get_zgyro(self): 
        if "AsZ(Â°/s)" in self.raw_data.columns.tolist():
            return self.raw_data["AsZ(Â°/s)"]
        else:
            return self.raw_data['AsZ(°/s)']