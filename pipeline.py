
import sys
import argparse
import pandas as pd
from processing import load_files, verbose
from viz import visualize
from features import metric_compare, calculate_gyro_metrics
import tsfel


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
    ),
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
    parser.add_argument(
        "-m",
        "--metrics",
        type=bool,
        help="Whether metrics should be calculated (any string input)",
        default=False # will not create output .csv if needed
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=bool,
        help="Whether descriptive statistics of the output table will be displayed to the log",
        default=False
    )
    
    return parser.parse_args()

def main():
    """Where the magic happens
    """
    
    args = parse_args()
    
    if args.fold == -1: #no folds
        dataset = load_files(args.input, fold = False) #generates Dataset of Zignals from the file.
    else:
        dataset = load_files(args.input, window_len=args.fold, fold = True)
        
    dataset.crop_sig() #crop all Zignals to the same signal length.
    
    if args.plot:
        visualize(dataset)
    
    if args.metrics:
        tsfel_names, accel_names = metric_compare(dataset)
        tsfel_names, gyro_names = calculate_gyro_metrics(dataset)

        #construct output table
        output_accel = pd.DataFrame(dataset.feature_table.values(), columns=accel_names + tsfel_names)
        output_gyro = pd.DataFrame(dataset.gyro_table.values(), columns=gyro_names + tsfel_names)

        if args.out:
            output_accel.to_csv(rf"out/acc_{args.out}")
            output_gyro.to_csv(rf"out/gyro_{args.out}")
            
        if args.verbose: #print descriptive statistics
            print("----- ACCELERATION -----\n\n")
            verbose(col_names=accel_names + tsfel_names, df=output_accel, start_metric="skew")
            print("\n\n----- GYROSCOPE -----\n\n")
            verbose(col_names=gyro_names + tsfel_names, df=output_gyro, start_metric="skew_gyro")
            
        return
    
if __name__ == "__main__":
    main()