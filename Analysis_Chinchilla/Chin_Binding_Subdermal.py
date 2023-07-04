# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 10:18:44 2023

@author: vmysorea
"""
###Chin subdermal electrode analysis to derive cortical ERPs
#%%Initialization
import sys
import os
sys.path.append('C:/Users/vmysorea/Anaconda3/python/')
#sys.path.append('C:/Users/vmysorea/OneDrive - purdue.edu/Documents/ANLffr-master/')
import warnings
import mne
import mne.io
import numpy as np
import anlffr
from anlffr.helper import biosemi2mne as bs
from matplotlib import pyplot as plt

import fnmatch
import pickle

#plt.matplotlib.use('TkAgg')
plt.switch_backend('QT5Agg')  # Making the plots interactive (Scrollable)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# Defining the dimensions and quality of figures
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 120

#%% Setting locs and loading data

froot = 'D:/PhD/Data/Chin_Data/Awake/Q414_Awake_EEG/'  # file location
save_loc='C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Chin_Analysis/Figures/'
pickle_loc = 'C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Chin_Analysis/Chin_Pickles/'

subjlist = ['Q414']  # Load subject folder
condlist = [1] #Coherence of 8 and 12 tones
condnames = ['12']

for subj in subjlist:
    evokeds = []
    # Load data and read event channel
    fpath = froot #+ subj + '/'
    bdfs = fnmatch.filter(os.listdir(fpath), subj +
                          '_Awake_Binding*.bdf')

    # Load data and read event channel
    rawlist = []
    evelist = []

    for k, rawname in enumerate(bdfs):
        rawtemp, evestemp = bs.importbdf(fpath + rawname, verbose='DEBUG',
                                         refchans=['EXG4', 'EXG5'], 
                                         exclude=['EXG6', 'EXG7', 'EXG8'])
        rawlist += [rawtemp, ]
        evelist += [evestemp, ]
    raw, eves = mne.concatenate_raws(rawlist, events_list=evelist)
    #raw, eves = raw.resample(4096, events=eves)
    raw.set_channel_types({'EXG1':'eeg'})
    raw.set_channel_types({'EXG2':'eeg'})
    raw.set_channel_types({'EXG3':'eeg'})
    #raw.set_eeg_reference('average', projection =True)

    # To check and mark bad channels

    raw.plot(duration=25.0, n_channels=37, scalings=dict(eeg=200e-6))

# %% Filtering
raw.filter(1., 15.)
raw.info
    
#%% Plotting Onset responses
picks = ['EXG2', 'EXG3']
for c, cond in enumerate(condlist):

    condname = condnames[c]
    epochs = mne.Epochs(
        raw, eves, cond, tmin=-0.3, proj=True,
        tmax=1.1, baseline=(-0.3, 0.0),
        reject=dict(eeg=150e-6))
    evoked = epochs.average()
    evokeds += [evoked, ]
    #picks = ['EXG3']
   #evoked.plot(titles='Onset Response - Q414 (Awake)')  #Add picks=picks to see one channel's response


epochs = mne.Epochs(raw, eves, event_id=[1], baseline=(-0.3, 0), proj=True,
                    tmin=-0.3, tmax=1.2, reject=dict(eeg=150e-6), picks=picks)
evoked = epochs.average()
#evoked.plot_topomap(0.3)
#picks = ['A8']
OnsetResponse_Total = evoked.plot (spatial_colors=True)
#%% Get subdermal data only 
pos = np.where(np.array(re_ref)==2)[0][0];
neg = np.where(np.array(re_ref)==1)[0][0];

big_mat = epochs.get_data(picks = sd_chans);

sig = big_mat[:,pos,:];
ref = big_mat[:,neg,:];

fin_sig = sig-ref;
m_fin_sig = np.mean(fin_sig,0);
std_fin_sig = np.std(fin_sig/np.sqrt(np.size(fin_sig,0)),0)

return m_fin_sig, std_fin_sig;
