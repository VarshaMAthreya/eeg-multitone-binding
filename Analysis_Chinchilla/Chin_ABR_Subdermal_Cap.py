# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 21:20:51 2023

@author: vmysorea
"""
import sys
import os
sys.path.append('C:/Users/vmysorea/Documents/mne-python/')
sys.path.append('C:/Users/vmysorea/Documents/ANLffr-master/')
import warnings
import mne
import numpy as np
from anlffr.helper import biosemi2mne as bs
from matplotlib import pyplot as plt

import fnmatch
import pickle

plt.switch_backend('QT5Agg')  # Making the plots interactive (Scrollable)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# Defining the dimensions and quality of figures
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 120

#%% Setting locs and loading data

froot = 'D:/PhD/Data/Chin_Data/Q415/'  # file location
save_loc='C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Chin_Analysis/Figures/'
pickle_loc = 'C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Chin_Analysis/Chin_Pickles/'

subjlist = ['Q415']  # Load subject folder

for subj in subjlist:
    evokeds = []
    # Load data and read event channel
    fpath = froot #+ subj + '/'
    bdfs = fnmatch.filter(os.listdir(fpath), subj +
                          '_Awake_ABR*.bdf')

    # Load data and read event channel
    rawlist = []
    evelist = []

    for k, rawname in enumerate(bdfs):
        rawtemp, evestemp = bs.importbdf(fpath + rawname, verbose='DEBUG',
                                         refchans=['EXG1', 'EXG2'])
        rawlist += [rawtemp, ]
        evelist += [evestemp, ]
    raw, eves = mne.concatenate_raws(rawlist, events_list=evelist)
    raw, eves = raw.resample(12288, events=eves)
    raw.set_channel_types({'EXG3':'eeg'})       #mastoid - positive peaks
    raw.set_channel_types({'EXG4':'eeg'})       #vertex - negative peaks
    raw.set_channel_types({'EXG5':'eeg'})       #ground
    #raw.set_eeg_reference('average', projection =True)

    # To check and mark bad channels

    raw.plot(duration=25.0, n_channels=41, scalings=dict(eeg=200e-6))

#%% Filtering 
for subj in subjlist:
    raw.filter(300., 6000.)
    raw.info

#%%Epoching ABRs
for subj in subjlist: 
    epochs = mne.Epochs(raw, eves, event_id=[1], baseline=(-0.002, 0.), proj=True,
                    tmin=-0.002, tmax=.015, reject=dict(eeg=50e-3), verbose = 'DEBUG')

    evoked = epochs.average()
    evoked_stderr = epochs.standard_error()
    evoked.plot()

#%%Get subdermal data and plot 
t=epochs[0].times
all_channels = (np.arange(1,32))
ep_mastoid = epochs.get_data()[:,36,:] #Mastoid -EXG3
ep_vertex= epochs.get_data()[:,35,:] #Vertex -EXG4
# ep_ground = epochs.get_data()[:,37,:] #Ground - EXG5
ep_all = evoked.data[all_channels,:]
ep_mean_all =ep_all.mean(axis=0)
ep_sem_all = ep_all.std(axis=0) / np.sqrt(ep_all.shape[0])

ep_subderm = ep_vertex - ep_mastoid #Inverting mastoid and non-inverting vertex 
ep_mean_subderm = ep_subderm.mean(axis=0)
ep_sem_subderm = ep_subderm.std(axis=0) / np.sqrt(ep_subderm.shape[0])

plt.plot(t*1000, ep_mean_subderm,label='Subdermal electrodes')
plt.fill_between(t*1000, ep_mean_subderm - ep_sem_subderm,
                     ep_mean_subderm + ep_sem_subderm,alpha=0.5)
plt.plot(t*1000, ep_mean_all,label='EEG Cap - 32 channels')
plt.fill_between(t*1000, ep_mean_all - ep_sem_all,
                      ep_mean_all+ ep_sem_all,alpha=0.5)
plt.xticks(ticks= [-2, 0, 2, 4, 6, 8, 10, 12, 14])
plt.ylim(-2.5*1e-6, 2.5*1e-6)
plt.title('Anesthetized ABR')
plt.legend()
plt.show()

plt.savefig(os.path.join(save_loc,subj + 'Anesthetized_ABR_CapvsSubdermal.png'),format='png', dpi=300)


    # #From Hari for EFR
    # x=epochs.get_data()[:,1,:]
    # plt.figure()
    # params = dict(Fs=raw.info['sfreq'], fpass=[100, 8000], tapers=[1, 1], itc=0)
    # y = x
    # plv, f = spectral.mtplv(y, params, verbose='DEBUG')
    # plt.plot(f, plv.T, linewidth=2)
    # plt.xlabel('Frequency (Hz)', fontsize=16)
    # plt.ylabel('Intertrial PLV', fontsize=16)
    # plt.title('EEG', fontsize=16)
    # plt.xlim([100, 5000])
    # plt.show()
    # #figname_eeg = subj + '_' + condstem + '_eeg-results.pdf'
    # #pl.savefig(fpath + figname_eeg)
    # y_eeg = y
