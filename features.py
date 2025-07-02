"""All manual features calculated for each Zignal
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from scipy.stats import gaussian_kde, entropy, skew, kurtosis
from scipy.signal import find_peaks, butter, sosfilt
import scipy.signal as sp
import csv
from sklearn.decomposition import PCA
import matplotlib.lines as mlines
import pywt 
import tsfel
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D 
from typing import List
from tqdm import tqdm
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq

from dataset import Dataset
from zignal import Zignal   


def kde(x: pd.Series):
    """Generates KDE object, which then has to be fed the output of the first argument

    Args:
        x (pd.Series): input signal

    Returns:
        np.ndarray, KDE object: series of 100 values between min and max value of x 
        to be inputted into KDE object.
    """
    return np.linspace(np.min(x), np.max(x), 100), gaussian_kde(x, bw_method="silverman")

def maxval(jerk: pd.Series) -> pd.Series:
    """Returns maximum absolute value of the MSJ metric. 

    Args:
        jerk (pd.Series): input signal

    Returns:
        pd.Series: Maximum absolute value of input signal.
    """
    return np.max(np.abs(jerk))

def jerk(data: pd.Series) -> pd.Series:
    """Returns the jerk (time derivative of acceleration) of the input signal. Presumes input signal is 
    magnitude of acceleration. Note: the second argument `pd.Series(data).index` might not be needed, only 
    np.gradient(data) is really doing something here.

    Args:
        data (pd.Series): input signal (should be acceleration)

    Returns:
        pd.Series: derivative of input signal
    """
    jk = pd.Series(np.gradient(data), pd.Series(data).index, name='jerk') #calculate jerk
    return np.nan_to_num(jk) #replace NaNs with zeros to avoid processing issues down the line.

def msj(data: pd.Series) -> pd.Series: 
    """Returns the Mean Squared Jerk metric. First, finds the derivative of input, squares it, 
    then takes the mean (self explanatory);

    Args:
        data (pd.Series): input signal (should be acceleration)

    Returns:
        pd.Series: Mean Squared value of the input signal.
    """
    return np.mean(np.square(jerk(data)))

def scaledmsj(data: pd.Series) -> pd.Series: 
    """Returns the scaled mean squared jerk of the signal. 
    Divided by the value of maximum value of the jerk, so MSJ gets mapped to 0-1 range.

    Args:
        jerk (pd.Series): input signal (should be acceleration)

    Returns:
        pd.Series: _description_
    """
    jk = jerk(data)
    return msj(data)/maxval(jk)

def moving_win(x: pd.Series, ws:int = 5) -> List[np.ndarray]: # Processing
    """Separates an input signal into non-overlapping windows of length ws.
    The average acceleration value, index locations, standard dev of acceleration values, and range of acceleration values
    are all returned.
    The subarrays are themselves not returned.

    Args:
        x (pd.Series): input signal
        ws (int, optional): window size, in INDICES. Defaults to 5. (meaning window with is 5 indices)

    Returns:
        List[np.ndarray]: avg. accel. value, index locations, st dev of accel., range of accel.
    """
    
    lo = 0
    hi = ws

    # declare arrays
    avgs = []
    idx = []
    ranges = []
    stds = []

    while hi < len(x):
        avgs.append(np.mean(x[lo:hi])) #used for average range
        idx.append((lo + hi) // 2)
        stds.append(np.std(x[lo:hi]))
        ranges.append(np.max(x[lo:hi]) - np.min(x[lo:hi])) #used for average range
        
        # new window begins where the prev one left off
        lo = hi 
        hi += ws
        
    return np.array(avgs), np.array(idx), np.array(stds), np.array(ranges)

def tot_acc(avgs_arr: np.ndarray) -> np.ndarray:
    """Returns the sum of the average acceleration calculated for each window. 
    Passes in the first array returned by moving_win()

    Args:
        avgs_arr (np.ndarray): array of average acceleration value for each window of Zignal

    Returns:
        np.ndarray: total acceleration metric.
    """
    return(np.sum(avgs_arr))

def cwt(data: pd.Series):
  """
  Calculates the magnitude or coeffecients of the continuous wavelet transform

   Args:
        dataset (Dataset): input Dataset of Zignals for pipeline run.

   Returns:
        coeffs: an array of cwt coeffecients
        freqs: an array of frequencies tied to the scales used in the transform
  """
  
  scales = np.arange(1, 100)
  coeffs, freqs = pywt.cwt(data, scales, 'cgau7', method="fft", sampling_period= 1/98)
  return coeffs, freqs

def fourier(data: pd.Series, zignal_fs):
    SUB_RATIO = 0.1 #used to subset the window we're plotting to get a closer look #separate
    #compute fft of signal 
    fft_out = np.abs(np.fft.fft(data)) #y axis
    freq = np.linspace(0, zignal_fs, len(fft_out)) #x axis <-- this is the confusing part
    freq_bin_subs = int(len(freq) * SUB_RATIO)
    j2 = 0
    intspikeindex = 0
    for c in range(0, len(fft_out)):
      if freq[c] < 0.1:
         intspikeindex = c
      if freq[c] > 10:
        j2 = c
        break
    if j2 != 0:
       max_fftfreq = freq[np.argmax(fft_out[intspikeindex+1:j2]) + intspikeindex + 1]
    else:
       max_fftfreq = freq[np.argmax(fft_out[intspikeindex+1:]) + intspikeindex + 1]
    return max_fftfreq

def apply_tsfel(signal: pd.Series) -> tuple[List, List]:
        cfg = tsfel.get_features_by_domain()  # This gets the default configuration of features
        extracted_features = tsfel.time_series_features_extractor(cfg, signal, fs=98, verbose = False) #removing progress bar here because it's muddying the output
        feature_name = []
        for i in extracted_features:
            feature_name.append(i)
        tsfel_vector = list(extracted_features.values)
        return tsfel_vector, feature_name

def kl_div(p: pd.Series, q: pd.Series) -> float: #Visualization
  """ Returns the KL divergence (entropy) between two distributions 
  
  Args:
        p (pd.Series): dist 1
        q (pd.Series): dist 2
  Returns:
        entropy(): entropy calculation from scipy
  """
  eps = 0.00001
  q = np.where(q == 0, eps, q) #replace any 0 values of q with eps, to avoid divide by zero
  return entropy(p, q) #element wise


def metric_compare(all_zignals: Dataset) -> pd.DataFrame:
    """Calculates a vector of features for every Zignal in the Dataset.
    Once all features are created, they're stored in `dataset.feature_table` and
    TSFEL features are stored in `dataset.tsfel_feature_table`.

    Args:
        dataset (Dataset): Input Dataset of Zignals for pipeline run.

    Returns:
        pd.DataFrame: Features for every Zignal in dataset. Shape is (num signals x num features).
    """
    # Initialize TSFEL feature storage if it doesn't exist
    if not hasattr(all_zignals, "tsfel_feature_table"):
        all_zignals.tsfel_feature_table = {}

    # Iterate through all signals and calculate features
    for z in tqdm(all_zignals.dataset, desc="Acceleration metrics calculation"):
        signal = z.enmoed  # Processed signal contained within the Zignal object

        # -- WHOLE SIGNAL METRICS --
        sk = skew(signal, bias=False)
        kurt = kurtosis(signal, bias=False, fisher=True)
        jk = jerk(signal)
        mean_sq_jerk = msj(signal)
        scaled_mean_sq_jerk = scaledmsj(signal)
        maxvalue = maxval(signal)
        tsfel_vector, tsfel_names = apply_tsfel(signal)

        # -- KDE METRICS --
        t, kde_obj = kde(signal)  # Generate KDE plot
        v = kde_obj(t)
        v = v/np.sum(v) 
        
        pdf_mean = np.sum(t*v) #find mean/center of pdf
        pdf_var = np.sum(v * (t - pdf_mean)**2) #find variance of pdf
        max_prob_index = np.argmax(v) 
        acc_max_prob = v[max_prob_index] #yield acceleration with max probailitity

        dommy_freqy = fourier(signal, z.fs) #frequency with maximum dominance

        # -- WINDOW METRICS --
        window_avgs, window_idx, window_stds, window_ranges = moving_win(signal, z.fs)
        tot_accel = tot_acc(window_avgs)
        
        avg_of_ranges = np.mean(window_ranges) #measuring "extremeness" in peaks - the average value of all the window ranges 
        avg_of_stds = np.mean(window_stds) #measuring variability in different points of the signal (average value of the window standard deviations)
        std_of_avgs = np.std(window_avgs) # measuring the how the window averages change across the signal

        #return all feature vec
        #constant list of all features. if you're adding featuers to feature.py, make sure to edit this too so
        #the output dataframe also works.
        COL_NAMES = ["participant", "date", "level", "fold", "trial_type", "trial_no", "fs", 
                       "skew", "kurt", "msj", "smsj", "maxval", "pdf_mean",
                       "pdf_var", "acc_max_prob", "tot_acc", "avg_of_ranges", "avg_of_stds",
                       "std_of_avgs", "fourier"] #ACCEL NAMES

        accel_feature_vec = [z.initials, z.date, z.level, z.fold, z.trial_type, z.trial_no, z.fs, 
                       sk, kurt, mean_sq_jerk, scaled_mean_sq_jerk, maxvalue, pdf_mean,
                       pdf_var, acc_max_prob, tot_accel, avg_of_ranges, avg_of_stds,
                       std_of_avgs, dommy_freqy]

        combined_array = np.append(accel_feature_vec, tsfel_vector) #with tsfel as well
        all_zignals.add_accel_feature_vec(z, combined_array)
        
    return tsfel_names, COL_NAMES

# Define a function to calculate the metrics for a given gyroscope axis (e.g., gyro_x, gyro_y, or gyro_z)
def calculate_gyro_metrics(all_zignals: Dataset):

    for z in tqdm(all_zignals.dataset, desc = "Gyroscope metrics calculation"): #loop through all the zignals in the dataset
        signal  = z.gyro #extract magnitude of gyro, with outliers removed

        filt_signal = bandpass_filter(signal, 1, 4, z.fs, order = 4)
        
        #notice they're mostly the same as above, but we're removing acceleration specific methods
        sk = skew(filt_signal, bias=False)
        kurt = kurtosis(filt_signal, bias=False, fisher=True)
        maxvalue = maxval(filt_signal)
        tsfel_vector, tsfel_names = apply_tsfel(filt_signal)

        # -- KDE METRICS --
        t, kde_obj = kde(filt_signal)  # Generate KDE plot, normalized
        v = kde_obj(t)
        v = v/np.sum(v) 
        
        pdf_mean = np.sum(t*v) #find mean/center of pdf, expected value
        pdf_var = np.sum(v * (t - pdf_mean)**2) #find variance of pdf
        max_prob_index = np.argmax(v) 
        acc_max_prob = v[max_prob_index] #yield acceleration with max probailitity

        dommy_freqy = fourier(filt_signal, z.fs) #frequency with maximum dominance
    
        std_dev = np.std(filt_signal) #std of plain gyro
        variance = np.var(filt_signal) #variance of plain gryo
        data_range = np.max(filt_signal) - np.min(filt_signal)
        rms = np.sqrt(np.mean(filt_signal**2)) #root mean square
        peak_to_peak = np.ptp(filt_signal) # Peak-to-Peak (Max - Min)
        autocorr = np.corrcoef(filt_signal[:-1], filt_signal[1:])[0, 1]  # Cross-correlation/auto (lag 1)
        zcr = np.count_nonzero(np.diff(np.sign(filt_signal))) # Zero Crossing Rate (ZCR)
        
        # Entropy (Shannon Entropy)
        histogram, bin_edges = np.histogram(filt_signal, bins=20, density=True)
        histogram = histogram[histogram > 0]  # Avoid log(0) --> #NOTE should probably use an epsilon here
        entropy = -np.sum(histogram * np.log(histogram))
        
        # Spectral Entropy (Requires FFT)
        freqs = np.fft.fftfreq(len(filt_signal))
        fft_vals = np.abs(np.fft.fft(filt_signal))
        psd = np.square(fft_vals)
        spectral_entropy = -np.sum(psd * np.log(psd + 1e-9))  # Adding small value to avoid log(0)
        
        # Dominant Frequency (from FFT)
        dominant_freq_idx = np.argmax(fft_vals)
        dominant_frequency = freqs[dominant_freq_idx]

        # List of variables (values)
        gyro_feature_names = ["participant", "date", "level", "fold", "trial_type", "trial_no", "fs", 
                       "skew_gyro", "kurt_gyro", "max_value_gyro", "pdf_mean_gyro", "pdf_var_gyro", "max_prob_index_gyro",
                       "acc_max_prob_gyro", "fourier_gyro", "std_dev_gyro", "var_gyro", "data_range_gyro", "rms_gyro",
                       "peak2_gyro", "autocorr_gyro", "zcr_gyro", "entropy_gyro", "spectral_entropy_gyro", "dominant_freq_idx_gyro", "dominant_frequency_gyro"]

        gyro_feature_vec = [z.initials, z.date, z.level, z.fold, z.trial_type, z.trial_no, z.fs,
                            sk, kurt, maxvalue, pdf_mean, pdf_var, max_prob_index, acc_max_prob, 
                            dommy_freqy, std_dev, variance, data_range, rms, peak_to_peak, autocorr,
                            zcr, entropy, spectral_entropy, dominant_freq_idx, dominant_frequency]

        combined_array = np.append(gyro_feature_vec, tsfel_vector) #with tsfel as well
        all_zignals.add_gyro_feature_vec(z, combined_array)

    return tsfel_names, gyro_feature_names

# Band-pass filter implementation (1-4 Hz)
def bandpass_filter(data, lowcut, highcut, zignal_fs, order=4):
    nyquist = 0.5 * zignal_fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y
