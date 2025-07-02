#utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from scipy.stats import gaussian_kde, entropy, skew, kurtosis
from scipy.signal import find_peaks, butter, sosfilt
import scipy.signal as sp
import seaborn as sns
import csv
from sklearn.decomposition import PCA
import matplotlib.lines as mlines
import pywt 
import tsfel
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D
import pywt

def load_data(path, names=None, skip=1):
  return (pd.read_csv(path))

#get raw measurements
def getAccX(df): return df["AccX(g)"]
def getAccY(df): return df["AccY(g)"]
def getAccZ(df): return df["AccZ(g)"]
def getAsX(df): return df["AsX(°/s)"]
def getAsY(df): return df["AsY(°/s)"]
def getAsZ(df): return df["AsZ(°/s)"]
#def getAngleX(df): return df["AngleX(°)"]
#def getAngleY(df): return df["AngleY(°)"]
#def getAngleZ(df): return df["AngleZ(°)"]

def mag(df): #Sensor
  try:
    return(np.sqrt(df["AccX(g)"] **2  + df["AccY(g)"] **2  + df["AccZ(g)"] ** 2))
  except:
    return(np.sqrt(df["Acceleration X(g)"] **2  + df["Acceleration Y(g)"] **2  + df["Acceleration Z(g)"] ** 2))

#makes all signals same length and removes outliers from all signals
def cwt(df):
  time = timestamp(df)[0]
  acc_mag = mag(df)
  scales = np.arange(1, 2000)
  coeffs, freqs = pywt.cwt(acc_mag, scales, 'cgau7', method="fft", sampling_period= 1/98)
  return time, coeffs, freqs

def crop_sig(*args): # Dataset

  min_len = np.inf #init
  cropped = []

  for arg in args:
    if len(arg) < min_len:
      min_len = len(arg)
  for arg in args:
    try:
      cropped.append(arg[0:min_len])
    except:
      print("invalid arrays passed. double check they aren't already cropped!")
  return cropped, min_len

#remove outliers, input number of deviations
def rm_outs(x, devs = 3): # Sensor
  x_mean = np.mean(x)
  x = np.where(x > np.mean(x) - devs * np.std(x), x, x_mean)
  x = np.where(x < np.mean(x) + devs * np.std(x), x, x_mean)
  return x

def enmo(x): # Sensor
  return mag(x) - 1

#return kde function
def kde(x): # Processing
    return (gaussian_kde(x, bw_method="silverman"))

#kullback-leiber divergence, p is "true dist", q is "sampled dist"
#pass in outputs of kernel density estimators
def kl_div(p, q): #Visualization
  eps = 0.00001
  q = np.where(q == 0, eps, q) #replace any 0 values of q with eps, to avoid divide by zero
  return entropy(p, q) #element wise

#for jenson-shannon divergence
def mixture(x): #Visualization
  pass

# TODO edit so that it's just returning the arrays themselves
def moving_win(x, ws = 5): # Processing 
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
    lo = hi
    hi += ws
  return [np.array(avgs), np.array(idx), np.array(stds), np.array(ranges)]

#pass in the first arg out of moving_win
def tot_acc(avgs_arr): # Manual
  return(np.sum(avgs_arr))

def jerk(data): #Manual
  jerk = pd.Series(np.gradient(data), pd.Series(data).index, name='jerk')
  return jerk

def msj(jerk): #Manual
  return np.mean(np.square(jerk))

def maxval(jerk): #Manual
  return np.max(np.abs(jerk))

def scaledmsj(jerk): #Manual
  return msj(jerk)/maxval(jerk)

def jerk_metrics(analysis):  #needed?
  jerk_data=[]
  msj_data=[]
  maxval_data=[]
  scaledmsj_data=[]
  count = 0
  for i in analysis: 
    jerk_data.append(jerk(i))
    for j in range(0, len(jerk_data[count])):
       if np.isnan(jerk_data[count][j]):
          jerk_data[count][j] = 0
    msj_data.append(msj(jerk(i)))
    maxval_data.append(maxval(jerk(i)))
    scaledmsj_data.append(scaledmsj(jerk(i)))
    count+=1

  return jerk_data, msj_data, maxval_data, scaledmsj_data

#TODO Finish implementing 

"""
def rythmicity(x, fs, start = 20, stop = 200):
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
  plt.title("rhythm")

  #return rhytmicity
  return np.std(np.diff(peaks))
"""

def decremental_effect(data): #Manual
    window_avgs, window_idx, window_std, window_range = moving_win(data, 200)
    low = 0
    high = 200
    window_count = 0
    peaks = []
    while high < len(data):
      for i in range(low, high):
          if data[i] >= (window_avgs[window_count] + 2*window_std[window_count]):
             peaks.append(i)
      high+= 200
      low += 200
      window_count+=1
    peaks_diffs = []
    for i in range(len(peaks) - 1):
      if np.isnan((data[peaks[i + 1]] - data[peaks[i]])/abs(data[peaks[i]]) * 100) == False:
        if data[peaks[i]] != 0:
            peaks_diffs.append((data[peaks[i + 1]] - data[peaks[i]])/abs(data[peaks[i]]) * 100)
        else:
            peaks_diffs.append((data[peaks[i + 1]] * 100))
      else:
         peaks_diffs.append(0)
    return peaks, np.mean(peaks_diffs)

"""
def wavlet(signal, time): #Manual
    scales = np.arange(1, 60)
    coeffs, freqs = pywt.cwt(signal, scales, 'cgau7', method = "fft", sampling_period=time)
    return freqs[coord[0]], coeffs, freqs, time
"""
    
#make sure to name your arguments (e.g. high_BK = signal_high_BK)
#will produce all metrics 
def metric_compare(data, name_arr, ws = 90): #Pipeline
  analysis = []
  fs_array = []
  #extract magnitude from the data tuples
  for tuple in data:
    df = tuple[4]
    mag_sig = rm_outs(enmo(df))
    analysis.append(mag_sig)
    fs_array.append(tuple[6])
  #cropped contains the cropped signals of all the input data
  analysis, min_len = crop_sig(*analysis)
  avgs = [] #array of arrays; each array element is an array of average values for a specific series within dataset
  idx = []
  ranges = []
  stds = []
  totalacc = []
  avgacc = []
  peak_sequence = []
  change_peaks = []
  wav_freqs = []
  wav_c = []
  wav_spec = [] 
  #wav_t  
  jerk_data, msj_data, maxval_data, scaledmsj_data = jerk_metrics(analysis)
  sk = [] #skew 
  kurt = [] #kurtosis
  mean_probs = []
  var_probs = []
  acc_maxp = []
  max_fftfreq=[]
  freq_bin_subs = []
  freq = []
  fft_out = []
  a_vals_kde = []
  t_lin_kde = []

  
  level = [] #bk or normal
  fold = [] #fold number. 0 if entire dataset, otherwise 1 through whatever the fold number goes up until.
  for i in data:
    data_pt = i
    #print(data_pt[2], data_pt[5])
    fold.append(data_pt[5])
    if "bk" in data_pt[2]:
      level.append(1)
    else:
      level.append(0)
  
  for i in range(0, len(analysis)):
     for j in range(0, len(analysis[i])):
        if np.isnan(analysis[i][j]):
           analysis[i][j] = 0
           
  #calculate metrics for moving window
  for i in range(len(analysis)):
    args = analysis[i]
    ##wav_freqs.append(wavlet(args,np.linspace(0,1,len(args)))
    sk.append(skew(args, bias = False)) #move to manual 
    kurt.append(kurtosis(args, bias = False, fisher= True)) # how much MORE normal?

    x_avgs, x_idx, x_stds, x_range = moving_win(args, ws = fs_array[i])
    avgs.append(x_avgs)
    idx.append(x_idx)
    ranges.append(x_range)
    stds.append(x_stds)

    totalacc.append(tot_acc(x_avgs))
    avgacc.append(np.mean(args))

    peak_indices, mean_percent_change_peaks = decremental_effect(args)
    change_peaks.append(mean_percent_change_peaks)
    peak_sequence.append(peak_indices)
    
    kde_data = kde(args) # additional
    t_lin = np.linspace(np.min(args), np.max(args), 100)
    a = kde_data(t_lin)
    a = a/np.sum(a) #--> turns to a probability
    a_vals_kde.append(a)
    t_lin_kde.append(t_lin)

    mean_probs.append(np.sum(t_lin * a))
    var_probs.append(np.sum(a * (t_lin - mean_probs[i])**2))
    max_ap_index = np.argmax(a)
    acc_maxp.append(t_lin[max_ap_index])

    SUB_RATIO = 0.1 #used to subset the window we're plotting to get a closer look #separate
    #compute fft of signal 
    fft_out.append(np.abs(np.fft.fft(args))) #y axis
    freq.append(np.linspace(0, fs_array[i], len(fft_out[i]))) #x axis <-- this is the confusing part
    freq_bin_subs.append(int(len(freq[i]) * SUB_RATIO))
    j2 = 0
    intspikeindex = 0
    for c in range(0, len(fft_out[i])):
      if freq[i][c] < 0.1:
         intspikeindex = c
      if freq[i][c] > 10:
        j2 = c
        break
    if j2 != 0:
       max_fftfreq.append(freq[i][np.argmax(fft_out[i][intspikeindex+1:j2]) + intspikeindex + 1])
    else:
       max_fftfreq.append(freq[i][np.argmax(fft_out[i][intspikeindex+1:]) + intspikeindex + 1])

    
        
    #wav_freqs.append(wavlet(args,np.linspace(0,1,len(args)))[0])
    #wav_c.append(wavlet(args,np.linspace(0,1,len(args)))[1])
    #wav_spec.append(wavlet(args,np.linspace(0,1,len(args)))[2])  
    #wav_t.append(wavlet(args,np.linspace(0,1,len(args)))[-1])
  
  #additional metrics
  avg_of_ranges = []
  for i in ranges:
    avg_of_ranges.append(np.mean(i))
  avg_of_stds = []
  for i in stds:
    avg_of_stds.append(np.mean(i))
  stds_of_stds = []
  for i in stds:
    stds_of_stds.append(np.std(i))
  stds_of_means = []
  for i in avgs:
    stds_of_means.append(np.std(i))
  
  splitted = [] #Signal
  for i in name_arr:
    splitted.append(i.split('_'))
  name = []
  date = []
  type2 = []
  trial_no = []
  for i in splitted:
       for j in range(0, len(i)):
          if j ==0:
             name.append(i[j])
          if j == 1:
             date.append(i[j])
          if j == 2:
             type2.append(i[j])
          if j == 3:
             trial_no.append(i[j])
  
  df1 = pd.DataFrame ({
       'Subj.': name,
       'Date': date,
       'Level': level,
       "Fold": fold,
       'Type_Trial': type2,
       'Trial #': trial_no,
       'Sampling Freq': fs_array,
       'Dominant Freq. of Mvmt.': max_fftfreq,
       'Acceleration with max-P.': acc_maxp,
       'Mean of PDF.': mean_probs,
       'Variance of PDF': var_probs,
    })     
  
  df2 = pd.DataFrame({
    'Var. in WMs': stds_of_means,
    'Mean of WSTDs': avg_of_stds,
    'Var. in WSTDs': stds_of_stds,
    'Average WRs': avg_of_ranges,
    'Average Acc.': avgacc,
    'Total Acceleration': totalacc,
  })

  df3 = pd.DataFrame({
    "Skew": sk, 
    "Kurtosis": kurt,
    'Mean % Change Between Adjacent Peaks': change_peaks,
    'Mean Squared Jerk': msj_data,
    'Max Jerk': maxval_data,
    'Scaled Mean Squared Jerk': scaledmsj_data,
  })


  #verbose
  #TODO make printing to terminal togglable
  print(df1)
  print(df2)
  print(df3)
  df4 = pd.concat([df2, df3], axis = 1)
  df5 = pd.concat([df1, df4], axis = 1)
  tsfel_matrix = tsfl(analysis, fs_array)
  #pca(tsfel_matrix, df5)
  df5.to_csv("bruh_v1.csv")
  return analysis, min_len, avgs, idx, ranges, stds, peak_sequence, jerk_data, a_vals_kde, t_lin_kde, freq_bin_subs, freq, fft_out

def process_name(file_name): #neeeded?
  file_name = file_name.split('.', 1)[0]
  if ("/" in file_name):
    parts = file_name.split("/")
  else:
    parts = file_name.split("\\")
  count = 0
  for i in file_name:
     if i == "/" or i == "\\":
        count+=1 #count number of slashes
        typeslash = i
  if len(parts) > count:
    if typeslash == "/":
      new_word = "/".join(parts[count:])  # Join parts after the second backslash
    else:
      new_word = "\\".join(parts[count:])  # Join parts after the second backslash
    return new_word
  
def process_th_dates(raw_date): #needed?
  date2 = str(raw_date).replace("-", "")
  date2 = date2.split(" ")[0]
  year = date2[2:4]
  date2 = date2[4:]
  date2 = date2 + "" + year
  return date2
   

#allows us to add any combination of text and csv files; will convert them and save them after checking that a converted version doesn't already exist
def load_files(path_file, window_len = 0, fs = 98, fold = False) -> list: #processing
    data_main = [] 
    name_arr = []
    with open(f"{path_file}", "r") as file2:
        for line in file2:
            fs = 98
            file_name2 = ""
            file_name = line.strip()
            file_name2 = ""
            df = pd.DataFrame()
            coded = ""
            with open(file_name, "r") as file:
              if (file_name[-4:] == "xlsx"):
                if os.path.isfile(file_name[:-4] + "csv"):
                   print("File already converted, please use csv next time.")
                   df = pd.read_csv(file_name[:-4] + "csv")
                   date2 = process_th_dates(str(df.iloc[0, 1]))
                   fs = 80
                   arr = process_name(file_name).split("_")
                   file_name2 = arr[1][2:] + "_" + date2 + "_" + "PDTapping_1"
                else:
                    df = pd.read_excel(file_name)
                    df = df.iloc[:, :6]
                    df.rename(columns={df.columns[3]: 'AccX(g)', 
                                  df.columns[4]:'AccY(g)', 
                                  df.columns[5]: 'AccZ(g)'}, inplace=True)
                    csv_file_name = file_name[:-4] + "csv"
                    df.to_csv(csv_file_name, index=False)  # index=False to avoid writing row indices  
                    date2 = process_th_dates(str(df.iloc[0, 1]))
                    arr = process_name(file_name).split("_")
                    file_name2 = arr[1][2:] + "_" + date2 + "_" + "PDTapping_1"
                    fs = 80
              elif (process_name(file_name).split("_")[0] == "PDMotion"):
                df = pd.read_csv(file_name)
                date2 = process_th_dates(str(df.iloc[0, 1]))
                fs = 80
                arr = process_name(file_name).split("_")
                file_name2 = arr[1][2:] + "_" + date2 + "_" + "PDTapping_1"
              else:  
                file_name2 = process_name(file_name)
                if file_name[-3:] == "txt":
                  if os.path.isfile(file_name[:-3] + "csv"):
                    file_name = file_name[:-3] + "csv" #name for saving purposes
                    print("File Already Converted Before, Please Use CSV Version")
                  else:
                    headers = file.readline().strip().split()[:-1]  # Split by comma and strip spaces
                    # Step 2: Read the data
                    data = []
                    data1 = []
                    g = 0
                    for line in file:
                      x = line.strip().split()
                      data.append(x)  # Split by space
                      data1.append(x[1:])
                      data1[g][0] = data[g][0] + " " + data[g][1]
                      g+=1
                    # Step 3: Create a DataFrame
                    df = pd.DataFrame(data1, columns=headers)
                    for column in df.columns:
                      if (column != "time" and column !=	"DeviceName" and column != "Version()"):
                        # Convert each column to float
                        df[column] = df[column].astype(float)
                    file_name = file_name[:-3] + "csv" #name for saving purposes
                    df.to_csv(file_name, index = False) 
                try:
                  df = pd.read_csv(file_name, index_col=False)
                except:
                  print(f"File {file_name} included in {path_file} does not exist in the data/ directory.")
            folds = fold_dfs(file_name2, df, window_len, fs, fold) 
            data_main.extend(folds) #note that extend only works by passing an iterable
            name_arr.extend([file_name2] * len(folds))
    return(data_main, name_arr)

# "fold" as in k-fold cross validation
def fold_dfs(file_name, df, window_len, fs = 98, fold = False) -> list: # Processing
  #NOTE: window_len is in seconds
  
  if file_name[-1] == ".": #remove
    file_name = file_name[:-1]
  
  collec = [] #collection of data points, either contains one df corresponding to the file, or multiple of the file if split is desired
  if fold == False:
    coded = file_name
    coded_spl = coded.split(sep = "_")
    #each item is a tuple (initial, date, level, trial, DataFrame, fold number = 0 if no split)
    data_pt = (coded_spl[0], coded_spl[1], coded_spl[2], coded_spl[3], df, 0, fs)
    #just append the one df (no folds)
    collec.append(data_pt)
    
  else: #fold desired
    
    #coded = file_name[:-4] #remove file extension from name (e.g. .csv)
    
    coded_spl = file_name.split(sep = "_")
    idx_len = window_len * fs #number of indicies in window (think about units!)
    num_folds = len(df.index) // idx_len #determine number of whole windows, integer division
    if num_folds < 2: #invalid folding
      print("The window length (in secs) and sampling frequency (in secs^-1) yields only one or zero folds. Please input valid combination.")
      return collec # error
    for i in range(num_folds):
      df_fold = df.iloc[i*idx_len:(i+1)*idx_len,:]
      data_pt = (coded_spl[0], coded_spl[1], coded_spl[2], coded_spl[3], df_fold, i+1, fs)
      collec.append(data_pt)
  
  #print(len(collec))
  return collec #list of tuples that correspond to data points, regardless of whether they 

# pass in dataframe from sensor,
# will return time in seconds since start & timedelta objects
def timestamp(df): #unneeded  
  STRING_FORMAT = "%Y-%m-%d %H:%M:%S:%f"
  string_times = df["time"]
  date_objs = string_times.apply(lambda x: datetime.strptime(x, STRING_FORMAT)) #convert all into datetime object
  t_deltas = date_objs - date_objs[0] #find differences between steps
  s = t_deltas.apply(lambda y: float(str(y.seconds) + "." + str(y.microseconds)))
  return s, t_deltas

# will graph metrics if called
def viz(data, name_arr, bin_size = 30): #main
    analysis, min_len, avgs, idx, ranges, stds, peak_sequence, jerk_data, a_vals_kde, t_lin_kde, freq_bin_subs, freq, fft_out = metric_compare(data, name_arr, ws = 90)
    
    order = []
    for i in name_arr:
      labels = i.split('_')
      label = labels[0] + " " + labels[2]
      if "long" in labels[2]:
        label += " " + labels[1][0:2] + "/" + labels[1][2:4]  #if trial no. same, use date to differentiate on labels
      else:
         label += labels[3]
      order.append(label)
    #plot histogram
    plt.figure(figsize=(10, 6))
    for i in range(len(analysis)):
        plt.hist(analysis[i], bins = 30, alpha = 0.2, 
                 label = order[i], density=True)
             
    plt.xlabel("Acceleration (g)")
    plt.ylabel("Frequency")
    plt.title(f"{data[0][0]} {data[0][1]} {data[0][2]}")
    plt.legend()
  
    #kde 
    plt.figure(figsize=(10, 6))
    for i in range(len(analysis)):
        plt.plot(t_lin_kde[i], a_vals_kde[i], label = order[i])

    plt.xlabel("Acceleration (g)")
    plt.ylabel("Probability (from KDE)")
    plt.title(f"{data[0][0]} {data[0][1]} {data[0][2]}")
    plt.legend()
    
    #plot line graph
    plt.figure(figsize=(10, 6))
    for i in range(len(analysis)):      
        plt.plot(np.arange(0, min_len, 1), analysis[i], 
                 label = order[i], alpha = 0.4) 
    plt.xlabel("Indices")
    plt.ylabel("Magnitude of Acceleration (g)")
    plt.title(f"{data[0][0]} {data[0][1]} {data[0][2]}")
    plt.legend()
    
    plt.figure(figsize=(10, 6))
    for i in range(len(analysis)):
        data_pt = data[i]
        plt.plot(freq[i][:freq_bin_subs[i]], fft_out[i][:freq_bin_subs[i]], 
                 label = order[i] ) #plot just indices for now...
    
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Coefficient")
    plt.title(f"{data[0][0]} {data[0][1]} {data[0][2]}")
    plt.legend()
        
    #kl-divergence heatmap
    kl_heat = [] #contains arrays of kl values
    kl_heat_names = [] #contains names of rows
    plt.figure(figsize = (12, 8))
    for i in range(len(analysis)): 
        kl_row = []
        
        kl_heat_names.append(order[i])
        
        kde_data = kde(analysis[i])
        t_lin = np.linspace(-3, 3, 100)
        plt.plot(t_lin, kde_data(t_lin), label = order[i])
        
        for j in range(len(analysis)): 
            kde_data_j = kde(analysis[j])
            t_lin_j = np.linspace(-3, 3, 100)
          
            kl = kl_div(kde_data(t_lin), kde_data_j(t_lin_j))
            kl_row.append(kl)
            
        kl_heat.append(kl_row)
    
    kl_df = pd.DataFrame(kl_heat)
    kl_df.columns = kl_heat_names
    kl_df.index = kl_heat_names
    plt.xlabel("Acceleration")
    plt.ylabel("KDE Values")
    plt.title(f"{data[0][0]} {data[0][1]} {data[0][2]}")
    plt.legend()

    
    plt.figure(figsize=(12, 8))
    sns.heatmap(kl_df, xticklabels=True, yticklabels=True, annot=True, cmap="viridis", vmin=0, vmax =1)
    plt.title("KL Divergence Heatmap")
    
    plt.figure(figsize=(10, 6))
    for i in range(len(analysis)):
       plt.plot(avgs[i], label = order[i])
    plt.xlabel("Indices")
    plt.ylabel("Window Means Over Time")
    plt.title(f"{data[0][0]} {data[0][1]} {data[0][2]}")
    plt.legend()

    plt.figure(figsize=(10, 6))
    for i in range(len(analysis)):
       plt.plot(stds[i], label = order[i])
    plt.xlabel("Indices")
    plt.ylabel("Window Variance Over Time")
    plt.title(f"{data[0][0]} {data[0][1]} {data[0][2]}")
    plt.legend()

    plt.figure(figsize=(10, 6))
    for i in range(len(analysis)):
       plt.plot(analysis[i][peak_sequence[i]], label = order[i])
    plt.xlabel("Indices")
    plt.ylabel("Window Peaks Over Time")
    plt.title(f"{data[0][0]} {data[0][1]} {data[0][2]}")
    plt.legend()

    plt.figure(figsize=(10, 6))
    for i in range(len(analysis)):
       plt.plot(jerk_data[i], label = order[i], alpha = 0.4)
    plt.xlabel("Indices")
    plt.ylabel("Jerk")
    plt.title(f"{data[0][0]} {data[0][1]} {data[0][2]}")
    plt.legend()

    plt.show()
    return


def tsfl(analysis, fs_array):
  cfg = tsfel.get_features_by_domain()
  matrix = []
  titles = []
  features = []
  titles = None
  j = 0
  for i in analysis: 
    features = tsfel.time_series_features_extractor(cfg, i, fs_array[j])
    matrix.append(features.values.flatten())
    if titles is None:
      titles = features.columns.tolist()
    j+=1
  matrix = pd.DataFrame(matrix, columns = titles)
  return matrix; 

def pca(tsfl_metrics, data_metrics):
   tsfl_metrics = tsfl_metrics
   data_metrics = data_metrics
   info = data_metrics[['Subj.', 'Date', 'Trial #', 'Type_Trial']]
   data_metrics.drop(columns = ['Subj.', 'Date', 'Trial #', 'Type_Trial'], inplace = True)
   data_metrics = data_metrics.fillna(0)
   tsfl_metrics = tsfl_metrics.fillna(0)
            
   df_combined = pd.concat([tsfl_metrics, data_metrics], axis = 1)
   n_components = 2
   pca = PCA(n_components = n_components)
   matrix_reduced = pca.fit_transform(df_combined)

   features = tsfl_metrics.columns.tolist() + data_metrics.columns.tolist()
   for i, component in enumerate(pca.components_[:2], start=1):
      top_indices=np.argsort(np.abs(component))[-5:][::-1]
      print("Top 5 contributing column indices")
      print(features[top_indices[0]])
      print(features[top_indices[1]])
      print(features[top_indices[2]])
      print(features[top_indices[3]])
      print(features[top_indices[4]])
   print(pca.explained_variance_ratio_)
  
   identity = []
   for i in range(len(info['Subj.'])):
      """
      if (info['Type_Trial'][i] == "PDTapping"):
         identity.append("PDTapping")
      else:
         identity.append(info['Subj.'][i] + info["Type_Trial"][i])
      """
      if (info['Type_Trial'][i] == "bkbrush"):
        identity.append('PD BRUSHING')
      else:
        identity.append(info['Subj.'][i] + info["Type_Trial"][i])
   groupings = list(dict.fromkeys(identity))
   unique_labels = list(set(identity)) 
   cmap = plt.cm.get_cmap('tab10', len(unique_labels))  
   label_colors = {label: cmap(i) for i, label in enumerate(unique_labels)}  # Map labels to colors

# Create a color list based on labels
   colors = [label_colors[label] for label in identity]
  
   plt.scatter(matrix_reduced[:, 0], matrix_reduced[:, 1], c = colors, cmap='viridis', edgecolor='k', s=100)
   plt.title('PCA Clustering Plot with Different Groups')
   plt.xlabel('Principal Component 1')
   plt.ylabel('Principal Component 2')

   legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=label_colors[label], markersize=10, label=label) for label in unique_labels]
   plt.legend(handles=legend_elements, title='Classes')
  # Show the plot
   plt.show()