from typing import List, Dict
from zignal import Zignal
import numpy as np
import pandas as pd

class Dataset():
    """Class that defines a Dataset object. Each Dataset is a collection of Zignals.
    Each run of the pipeline analyzes a Dataset, and will calcualte/visualize all the Zignals
    in the Dataset.
    """
    
    def __init__(self):
        """Initializes empty Dataset.
        """
        self.dataset: List[Zignal] = [] #empty dataset

        self.feature_table: Dict[Zignal, List[float]] = {} #dictionary that has key value pairs of Zignal: feature_list
        self.gyro_table: Dict[Zignal, List[float]] = {} #same as feature_table but has gyro  data

        self.tsfel_feature_table_columns: List[str]=[]
        self.tsfel_feature_table: Dict[Zignal, List[float]]={}

    def add_zignal(self, s: Zignal):
        """Expands the Dataset by adding a new Zignal.

        Args:
            s (Zignal): input zignal to be added
        """
        self.dataset.append(s)
    
    def get_size(self):
        size = 0 #to count number of zignals
        for i in self.dataset:
            size+=1
        return size
    
    
    def get_zignals(self) -> List[Zignal]:
        """Returns all Zignals in Dataset.

        Returns:
            List[Zignal]: Dataset.
        """
        return self.dataset
    
    def add_accel_feature_vec(self, s: Zignal, vec: List[float]):
        
        self.feature_table[s] = vec #update feature table dictionary.
        
        return

    def add_gyro_feature_vec(self, s: Zignal, vec: List[float]):

        self.gyro_table[s] = vec 

        return
    
    def crop_sig(self):
        """Crops all Zignals to the shortest ones in the Dataset.
        """
        
        min_len = np.inf #initialize min len as highest possible value
        for s in self.dataset: #find min length
            if len(s.enmoed) < min_len: 
                min_len = len(s.enmoed)
                    
        for s in self.dataset: #crop the signals
            try:
                s.enmoed = s.enmoed[0:min_len]
            except:
                print("invalid array's passed. double check they aren't empty (or cropped already).")
                
            
         
    