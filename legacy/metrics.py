import os
import sys
import argparse
import pandas as pd
import pip 
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from scipy.stats import gaussian_kde
import scipy.signal as sp
'''
 

# def install(package):
#    if hasattr(pip, 'main'):
#       pip.main(['install', package])
#    else:
#        pip._internal.main(['install', package])
        
#install("pywt")

from utils import load_files, viz, metric_compare
        
def parse_args():
    """
    Parse command-line arguments.

    Returns:
        args (an argparse.Namespace): Stores command-line attributes
    """
    #initialize
    parser = argparse.ArgumentParser(description="generate metrics for input signals")
    #arguments
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to txt file with paths along with names ",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        help="Path to csv file with output metrics",
        default="out.csv",
    )
    parser.add_argument(
        "-f",
        "--fold",
        type=int,
        help="Trigger folding of input data frames or not. Pass in window length in SECONDS.",
        default=-1
    )
    parser.add_argument(
        "-p",
        "--plot",
        type=bool,
        help="Whether plots should be generated (any string input)",
        default=False,
    )
    return parser.parse_args()


#main loop
def main():
    pd.set_option('display.width', None)  # Adjust to terminal width
    args = parse_args()
    if args.fold == -1: #fold not needed
        data, data_name = load_files(args.input, fold=False) #pass in data file, along with whether or not to split
    else:
        data, data_name= load_files(args.input, window_len=args.fold, fold=True)
    if args.plot:
        viz(data, data_name)
    else: #TODO FIX issues with what metric_compare returns
        df = pd.DataFrame(metric_compare(data, data_name, ws = 90)[0]) 
        df.to_csv(rf"out/{args.out}") broken
    return


# if command-line run, call main
if __name__ == "__main__":
    main()


















