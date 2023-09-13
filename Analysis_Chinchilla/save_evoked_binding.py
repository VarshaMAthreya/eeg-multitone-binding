# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 11:28:45 2023

@author: vmysorea
"""

### Analyze the onset differences between anesthesia protocols -- Binding Only -- First, let's save evoked onsets 

import sys
sys.path.append('C:/Users/vmysorea/mne-python/')
# sys.path.append('C:/Users/vmysorea/Documents/ANLffr/')
sys.path.append('C:/Users/vmysorea/ANLffr-master/')
import warnings
import mne
import numpy as np
from anlffr.helper import biosemi2mne as bs
from matplotlib import pyplot as plt
import os
import fnmatch
from scipy.io import savemat

plt.switch_backend('QT5Agg')  # Making the plots interactive (Scrollable)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# Defining the dimensions and quality of figures
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 120

#%% Get BDF 

froot = 'F:/PhD/Data/Chin_Data/Anesthetized/'  # file location
save_loc='F:/PhD/Data/Chin_Data/AnalyzedBinding_matfiles/'

subjlist = ['Q417']  # Load subject folder
condlist = [1] 
condnames = ['12']

for subj in subjlist:
    evokeds = []
    # Load data and read event channel
    fpath = froot + subj + '/'
    bdfs = fnmatch.filter(os.listdir(fpath), subj +'_Anesthetized_Binding*.bdf')

    # Load data and read event channel
    rawlist = []
    evelist = []

    for k, rawname in enumerate(bdfs):
        rawtemp, evestemp = bs.importbdf(fpath + rawname, verbose='DEBUG',
                                         refchans=['EXG1', 'EXG2'])
        rawlist += [rawtemp, ]
        evelist += [evestemp, ]
    raw, eves = mne.concatenate_raws(rawlist, events_list=evelist)
    raw, eves = raw.resample(4096, events=eves)
    raw.set_channel_types({'EXG3':'eeg'})           #Mastoid -34
    raw.set_channel_types({'EXG4':'eeg'})           #Vertex -35
    raw.set_channel_types({'EXG5':'eeg'})           #Ground -36
    raw.info['bads'].append('EXG6') 
    raw.info['bads'].append('EXG7')
    raw.info['bads'].append('EXG8')
    raw.info['bads'].append('A27') 
    
    # if subj == 'Q414_1':
    #     raw.set_channel_types({'EXG2':'eeg'})           #Mastoid - 35
    #     raw.set_channel_types({'EXG3':'eeg'})           #Vertex - 34
    #     raw.set_channel_types({'EXG1':'eeg'})           #Ground - 33
        
    #raw.set_eeg_reference(ref_channels='average')

    # To check and mark bad channels
    # raw.plot(duration=25.0, n_channels=41, scalings=dict(eeg=200e-6))

#%% Marking bad chans
    if subj == 'Q417':
        raw.info['bads'].append('A1')
        raw.info['bads'].append('A12')
        raw.info['bads'].append('A26')
    
#%% Filtering 
    raw.filter(1., 30.)
    raw.info
    
#%% Plotting Onset responses
    epochs = mne.Epochs(raw, eves, event_id=[1], baseline=(-0.3, 0), proj=True,
                        tmin=-0.3, tmax=1.2, reject=dict(eeg=200e-6))
    evoked = epochs.average()
    OnsetResponse_Total = evoked.plot(spatial_colors=True) #Butterfly plots
    # times = np.linspace (0.15, 0.3,5)
    # # times = np.linspace (1.15, 1.3,5)
    # evoked.plot_topomap(times=times)
    
    ##Plotting full time
    epochs = mne.Epochs(raw, eves, event_id=[1], baseline=(-0.3, 0), proj=True,
                        tmin=-0.3, tmax=5.5, reject=dict(eeg=200e-6))#, picks=picks)
    evoked = epochs.average()
    #evoked.plot_topomap(0.3)
    #picks = ['A8']
    OnsetResponse_Total = evoked.plot (spatial_colors=True, picks=['A18', 'A2', 'A3', 'A23', 'A8','A22', 'A7'])   ##EXG3(Mastoid)-Negative waves;EXG4-Positive(Follows cap direction)
    times = np.linspace (1.15, 1.3,5)
    evoked.plot_topomap(times=times, show_names=True)
    
    #Plotting mean of all cap channels
    t=epochs[0].times
    
    all_channels = (np.arange(1,33))
    ep_all = evoked.data[all_channels,:]
    ep_mean_all =ep_all.mean(axis=0)
    ep_sem_all = ep_all.std(axis=0) / np.sqrt(ep_all.shape[0])
    
    plt.plot(t, ep_mean_all,label='EEG Cap - 32 channels')
    plt.fill_between(t, ep_mean_all - ep_sem_all,
                          ep_mean_all+ ep_sem_all,alpha=0.5)
    #plt.xticks(ticks= [-2, 0, 2, 4, 6, 8, 10, 12, 14])
    #plt.ylim(-2.5*1e-6, 2.5*1e-6)
    plt.title('Binding Awake')
    plt.legend()
    plt.show()