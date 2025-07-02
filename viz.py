"""Visualizes all Zignals in a Dataset.
"""

from dataset import Dataset
from features import kde, cwt, kl_div
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List

def kl_plot(dataset: Dataset) -> pd.DataFrame:
    """Returns a DataFrame with the 2d square matrix of KL divergences.

    Args:
        dataset (Dataset): input dataset to visualize
    
    Returns:
        heatmap (pd.DataFrame): 2d square matrix of kl divs
    """
    all_zignals = dataset
    kl_row, kl_col = [], []
    t = np.linspace(-5, 5, 300)
    for i in all_zignals.dataset:
        for j in all_zignals.dataset:
            kde_eq_i = kde(i.enmoed)[1] #calculate kde equation
            kde_eq_j = kde(j.enmoed)[1]
            kl_col.append(kl_div(kde_eq_i(t), kde_eq_j(t))) #return entropy
        kl_row.append(kl_col)
        kl_col = []
    return pd.DataFrame(kl_row)

def visualize(dataset: Dataset):
    all_zignals = dataset
    
    #warning so people's computers don't get messed up lol
    if len(all_zignals.dataset) > 15:
        opt = input("WARNING: You are visualizing more than 15 signals, each of which have multiple plots. Many windows will open. Continue [y/n]?").lower().strip()
        if opt == "n":
            return
            
    ##---HISTOGRAM PLOT---
    plt.figure(figsize=(10, 6))
    for z in all_zignals.dataset:
        sig_name = str(z) #convert the Zignal to a string; managed by __str__ argument
        plt.hist(z.enmoed, bins = 30, alpha = 0.2, 
                 label = sig_name, density=True)
    
    
    #---KDE PLOT--- #no probability yet...
    plt.figure(figsize=(10, 6))
    for z in all_zignals.dataset:
        sig_name = str(z)
        t, kde_obj = kde(z.enmoed)
        v = kde_obj(t)
        plt.plot(t, v, label = sig_name)
    
    plt.legend()
    plt.show()
    
    #--- KL DIV PLOT ---
    plt.figure(figsize=(12, 8))
    heat_map = kl_plot(all_zignals)
    names = [str(zig) for zig in all_zignals.dataset]
    heat_map.index = names
    heat_map.columns = names
    sns.heatmap(heat_map, xticklabels=True, yticklabels=True, annot=True, cmap="viridis", vmin=0, vmax =1)
    plt.xlabel("Acceleration")
    plt.ylabel("KDE Values")
    plt.show()


    for z in all_zignals.dataset:
        # Create a new figure for each dataset
        fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(17, 9), sharex=True)
        
        sig_name = str(z)

        component_sensor_types = [["g", "g", "g", "g"],
                                ["degrees", "degrees", "degrees", "degrees"]]

        component_data_names = [
            [f"{sig_name} AccX", f"{sig_name} AccY", f"{sig_name} AccZ", f"{sig_name} AccMag"],
            [f"{sig_name} GyroX", f"{sig_name} GyroY", f"{sig_name} GyroZ", f"{sig_name} GyroMag"]
        ]
        
        component_data = [
            [z.xa, z.ya, z.za, z.enmoed], 
            [z.xg, z.yg, z.zg, z.gyro]
        ]
        
        # Loop through and plot data on each subplot
        for c in range(2):
            for r in range(4):
                ax[r, c].plot(component_data[c][r])

                ax[r, c].set_title(f"{component_data_names[c][r]}")
                ax[r, c].set_xlabel("Index")
                ax[r, c].set_ylabel(component_sensor_types[c][r])

        # Show the figure
        plt.tight_layout()  # This adjusts the spacing between subplots to avoid overlap
        plt.show()  # This renders the figure

    """
    #--- tSNE PLOT --- #note, need to also generate metrics with the -m flag, so that the feature table gets updated

    all_features = pd.DataFrame(all_zignals.feature_table.values()).iloc[:,7:] #only accel features for now

    tsne = TSNE(n_components=2, perplexity= len(all_features)-1 , random_state = 783)
    X_dim_reduc = tsne.fit_transform(all_features)

    plt.figure()
    plt.scatter(X_dim_reduc[:, 0], X_dim_reduc[:, 1], s = 10)
    plt.xlabel("tSNE 1")
    plt.xlabel("tSNE 2")
    plt.title("tSNE viz")
    plt.show()

    #--- CWT PLOT ---  
    plt.figure(figsize=(10, 6))
    
    cwt_data = []
    cwt_data_labels = [] 
    for z in all_zignals.dataset:
       sig_name = str(z)
       c, f = cwt(z.enmoed)
       
       time = np.linspace(0,len(z.enmoed))
       plt.imshow(abs(c), extent=[time[0], time[len(time)-1], f[-1], f[0]],
             interpolation="bilinear", cmap="jet", aspect="auto",
             vmax=abs(c).max(), vmin=-abs(c).max())
       plt.gca().invert_yaxis()
       plt.colorbar()
       plt.ylabel("Scale")
       plt.xlabel("Time")
       plt.title(f'Continuous Wavelet Transform (Adaptive) - {sig_name}')
       plt.show()
       
       cwt_data.append(c)
       cwt_data_labels.append(int(z.level)) #0 healthy vs 1 control
    
    np.save(rf"cwt_data/cwt_dataset.npy", cwt_data)
    np.save(rf"cwt_data/cwt_dataset_labels.npy", cwt_data_labels)
"""

