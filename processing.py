"""The processing.py file includes general utilities for processing incoming data.
    Such as:
    - load_files: takes in an input .txt file OR directory and loads in all files contained in them
    - process_file: generates a Zignal from every input file path

""" 

import glob
import os
import numpy as np
import pandas as pd
from zignal import Zignal
from dataset import Dataset

def load_files(path: str, window_len: int = 7, fs: int = 98, fold: bool = False) -> Dataset:
    """Loads all files staged for analysis in this pipeline run.

    Args:
        path (str): path to a DIRECTORY or path to a TXT file with data trials
        window_len (int, optional): Fold length in seconds. Defaults to 0.
        fs (int, optional): Sampling Frequency. Defaults to 98.
        fold (bool, optional): Include Folds? Defaults to False.

    Returns:
        dataset (Dataset): Collection of all the Zignals
    """
    dataset = Dataset()
    try:
        if os.path.isdir(path): #check if directory. if yes, analyze all the signal files within the directory
            leaves = [file for file in glob.glob(f"{path}/**/*", recursive=True) if os.path.isfile(file)] #return all files within the folder
            for leaf in leaves: #each "leaf" is a file of the file tree 
                file_extension = os.path.splitext(os.path.basename(leaf))[1]
                if file_extension == ".txt": #handle txt files
                    df = pd.read_csv(leaf, sep = "\t", encoding_errors="ignore")
                    if len(df) < (98 * 30): 
                        continue
                elif file_extension == ".csv":
                    df = pd.read_csv(leaf, encoding_errors="ignore")
                    if len(df) < (98 * 30):
                        continue
                
                if fold:
                    fold_dfs(df, dataset, leaf, window_len, fs)
                else: #just read in file
                    sig = process_file(df, leaf)
                    dataset.add_zignal(sig)
                    
        elif os.path.isfile(path): #if file, analyze every file in the line
            with open(path, "r") as file_collection:
                for file_path in file_collection: #notice that each line is a separate file path
                    file_extension = os.path.splitext(os.path.basename(file_path))[1]
                    if ".txt" in file_extension: #handle txt files
                        df = pd.read_csv(file_path.strip(), sep = "\t", encoding_errors="ignore")
                        if len(df) < (98 * 30): 
                            continue
                    elif ".csv" in file_extension:
                        df = pd.read_csv(file_path.strip(), encoding_errors="ignore")
                        if len(df) < (98 * 30): 
                            continue
                    if fold:
                        fold_dfs(df, dataset, file_path.strip(), window_len, fs)
                    else: #just read in file
                        sig = process_file(df, file_path.strip())
                        dataset.add_zignal(sig)
                        
    except FileNotFoundError:
        print("The file you passed was not in the directory.")
        raise   
    
    return dataset

def process_file(df: pd.DataFrame, file_path: str, fold_no: int = 0) -> Zignal:
    """Generates Zignal from input file.
    
    [ONLY WORKS WITH CSVs FOR NOW #TODO add .txt support]

    Args:
        df (pd.DataFrame): raw data frame
        file_path (str): path to singular trial of WITMOTION
        fold_no (int): fold number. 0 if we include the entire trial, vs. 1-n for n folds. Defaults to n.

    Returns:
        s (Zignal): returns Zignal from files
    """
    with open(f"{file_path}", "r") as file:
        basefile = os.path.basename(file_path) #finds file name and removes extensions. i.e. "sup\bruh.txt" --> "bruh.txt"
        trimmed = os.path.splitext(basefile)[0] #removes extension "bruh.txt" --> "bruh"
        info = trimmed.strip().split("_")
        s = Zignal(raw_data=df,  #create Signal object
                   initials=info[0], 
                   date=info[1], 
                   trial_type=info[2], 
                   trial_no=info[3],
                   fold = fold_no, 
                   sampling_frequency=98)
        
    return s #return signal associated with this file

def fold_dfs(df: pd.DataFrame, dataset: Dataset, file_name: str, window_len: int = 7, fs: int = 98):
    """Creates smaller subset trials (called folds, from k-fold cross val.) and adds them to the Dataset.
    The subset trials are signals of the specified window_len, which is argument that takes in the number of seconds
    the trial should be. 

    Args:
        df (pd.DataFrame): raw df that was loaded from the file
        dataset (Dataset): dataset for this run.
        file_name (str): file path for this file
        window_len (int): window length in SECONDS. Defaults to 7 (secs).
        fs (int, optional): sampling frequency. Defaults to 98.
    """
    idx_len = window_len * fs #number of indicies in window (think about units!)
    num_folds = len(df.index) // idx_len #determine number of whole windows, integer division
    
    if num_folds < 1: #invalid folding
      print("The window length (in secs) and sampling frequency (in secs^-1) yields only one or zero folds. Please input valid combination.")
      print(file_name + " was removed from analysis due to insufficient size")
    else:
        for i in range(num_folds):
            df_fold = df.iloc[i*idx_len:(i+1)*idx_len,:]
            sig = process_file(df_fold, file_name, fold_no=i+1) #pass in the smaller df, but retain the same info from file name
            dataset.add_zignal(sig)

def verbose(col_names: list, df: pd.DataFrame, start_metric: str) -> None:
    table_max_width = max([len(feature_name) for feature_name in col_names])
    PADDING = 2
    assert table_max_width > 0
    
    first_feature_loc = df.columns.get_loc(start_metric) #contains name of first real feature
    
    #cut off all metadata
    sub_df = df.iloc[:,first_feature_loc:].astype(float)
    sub_columns = col_names[first_feature_loc:]
    
    #calc desc stats
    mus = sub_df.mean(axis=0) #downwards
    sigmas = sub_df.std(axis=0)

    print(f"{'Feature':<{table_max_width+PADDING}} | {'Mean':>{10}} | {'Std. Dev.':>{10}}")
    print("-" * (table_max_width+PADDING) + "-" * 25)
    for i in range(len(sub_columns)):
        print(f"{sub_columns[i]:<{table_max_width+PADDING}} | {mus.iloc[i]:>10.5f} | {sigmas.iloc[i]:>10.5f}")

    return

            
        
