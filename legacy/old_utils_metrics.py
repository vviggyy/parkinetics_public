#utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sp
from datetime import datetime


def load_data(path, names=None, skip=1):
  return (pd.read_csv(path))

def getAccX(df):
  return df["AccX(g)"]

def getAccY(df):
  return df["AccY(g)"]

def getAccZ(df):
  return df["AccZ(g)"]

def getAsX(df):
  return df["AsX(°/s)"]

def getAsY(df):
  return df["AsY(°/s)"]

def getAsZ(df):
  return df["AsZ(°/s)"]

def getAngleX(df):
  return df["AngleX(°)"]

def getAngleY(df):
  return df["AngleY(°)"]

def getAngleZ(df):
  return df["AngleZ(°)"]

def mag(df):
  return(np.sqrt(df["AccX(g)"] **2  + df["AccY(g)"] **2  + df["AccZ(g)"] ** 2))

def ft(x):
  coeff = np.fft.fft(x)
  return [np.real(coeff), np.imag(coeff), coeff]

## -- used if data is imported as strings --
def filter_str(val):
  return(isinstance(val, str))


def remove_str(df):
  return(df[~df.applymap(filter_str)])

# calculate welch periodogram
# then min-max normalize between 0-1
def normalized_psd(x, fs = 20):
  freqs, spec = sp.signal.welch(x, fs = fs)
  norm_spectra = (spec - np.min(spec))/np.max(spec)
  return np.array([freqs, norm_spectra])

#takes normalized PSD, creates a frequency distribution
def create_frequency_distribution(freq, spectra):
  sum = spectra.sum();
  for i in range(0, len(spectra)):
    spectra[i] = spectra[i]/sum;
  return freq, spectra;

#determines parameters for bandpass filter to remove high freq noise
def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

# Function to apply the high-pass filter to accelerometer data
def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

## Takes in the normalized PSD signal, finds peaks,
#locates frequencies associated with those peaks,
#returns the average of those peaks.
# average dominant frequency
def peaks(freq_arr, norm_array, h = 0.8, prom = 0.4):
  pks, _ = sp.signal.find_peaks(norm_array, height = h, prominence = prom) #returns indices of peaks
  freq_pks = freq_arr[pks] #contains frequencies associated with the peaks

  return np.mean(freq_pks)

def average_value_trapezoidal_rule(freq_arr, norm_array):
  sum_of_norm_array=sum(norm_array)
  #print(sum_of_norm_array)
  #print(norm_array)
  normalized_norm_array=norm_array/sum_of_norm_array
  #print(normalized_norm_array)
  average=0
  for i in range(0, len(normalized_norm_array)):
    average+=normalized_norm_array[i]*freq_arr[i]
  return average

def average_value_simpsons_rule(freq_arr, norm_array):
  return simps(norm_array, freq_arr)

# Return True/False output
def bk_compare(df1, df2, label1 = "l1", label2 = "l2"):
  mag_1 = mag(df1)
  mag_2 = mag(df2)

  norm_1 = normalized_psd(mag_1)
  norm_2 = normalized_psd(mag_2)

  sig_1_freq = peaks(norm_1[0], norm_1[1])
  sig_2_freq = peaks(norm_2[0], norm_2[1])

  # TODO plot
  # TODO fix

  #returns array
  #true if df1 signal has higher frequency than others
  #sig_1_freq, avg dominant freq
  #sig_2_freq, avg dominant freq
  return (np.array([sig_1_freq > sig_2_freq, sig_1_freq, sig_2_freq]))


  def three_subplot(sig1, sig2, sig3,
                  title1 = "title3",
                  title2 = "title3",
                  title3 = "title3",
                  x_lim = (0, 400),
                  y_lim = (0, 3),
                  fs = 5,
                  x_ax = "Time (s)",
                  y_ax = "Magnitude of Acceleration"):

  fig, ax = plt.subplots(3, 1)

  ax[0].plot(idx_to_time(sig1, fs), sig1)
  ax[0].set_ylim(y_lim[0], y_lim[1])
  ax[0].set_xlim(x_lim[0], x_lim[1])
  ax[0].set_title(title1)

  ax[1].plot(idx_to_time(sig2, fs), sig2)
  ax[1].set_ylim(y_lim[0], y_lim[1])
  ax[1].set_xlim(x_lim[0], x_lim[1])
  ax[1].set_title(title2)

  ax[2].plot(idx_to_time(sig3, fs), sig3)
  ax[2].set_ylim(y_lim[0], y_lim[1])
  ax[2].set_xlim(x_lim[0], x_lim[1])
  ax[2].set_title(title3)

  fig.text(0.5, 0.04, x_ax, ha='center', va='center')
  fig.text(0.02, 0.5, y_ax, ha='center', va='center', rotation='vertical')

  plt.subplots_adjust(wspace=0.7)
  plt.subplots_adjust(hspace=0.7)

## TODO add degravify (AS)

#calculates single metric (rhytmicity). 
#std. of distance between peaks.
#may need to adjust butterworth filter, tested on dummy data
#prominence is hard coded
#df - 9axis data
#fs - sampling frequency
#start, stop - ints that signify how you want to subset the data.
def rythmicity(df, fs = 10, start = 20, stop = 200):
  RANGE = np.array(range(start, stop))

  #apply butterworth filter
  sos = sp.signal.butter(1, 2, 'lp', fs=fs, output='sos')
  filt = sp.signal.sosfilt(sos, mag(df))

  #subset and find peaks
  filt = filt[start:stop]
  peaks, properties = find_peaks(filt, prominence=0.015)

  #plotting 
  t = RANGE / fs

  plt.plot(t, filt)
  plt.plot(t[peaks], filt[peaks], "x", color = "blue")

  #return rhytmicity
  return np.std(np.diff(peaks))
  
  def timestamp(df):
    STRING_FORMAT = "%Y-%m-%d %H:%M:%S:%f"
    string_times = df["time"]
    date_objs = string_times.apply(lambda x: datetime.strptime(x, STRING_FORMAT)) #convert all into datetime object
    t_deltas = date_objs - date_objs[0] #find differences between steps
    s = t_deltas.apply(lambda y: float(str(y.seconds) + "." + str(y.microseconds)))
    return s, t_deltas
