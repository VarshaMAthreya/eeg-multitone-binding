# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:05:06 2023

@author: vmysorea
"""
import sys
import os
sys.path.append('C:/Users/vmysorea/Documents/mne-python/')
sys.path.append('C:/Users/vmysorea/Documents/ANLffr-master/')
from scipy.io import savemat
import warnings
import mne
import numpy as np
from anlffr.helper import biosemi2mne as bs
from matplotlib import pyplot as plt
import fnmatch
from mne.time_frequency import (tfr_multitaper) 

plt.switch_backend('QT5Agg')  # Making the plots interactive (Scrollable)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# Defining the dimensions and quality of figures
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 120

#%% Setting locs and loading data

froot = 'D:/PhD/Data/Chin_Data/Q419_Anes_Awake_GDT+Binding/'  # file location
save_loc_mat='C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Chin_Analysis/Freq_Band/'
save_loc_fig='C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Chin_Analysis/EFR/Figures/'

#%%Evoked Response -- Get EFR 
    
subjlist = ['Q419']  # Load subject folder

for subj in subjlist:
    # Load data and read event channel
    fpath = froot #+ subj + '/'
    bdfs = fnmatch.filter(os.listdir(fpath), subj +
                          '_Anes_GDT.bdf')

    # Load data and read event channel
    rawlist = []
    evelist = []

    for k, rawname in enumerate(bdfs):
        rawtemp, evestemp = bs.importbdf(fpath + rawname, verbose='DEBUG',
                                         refchans=['EXG1', 'EXG2'])
        rawtemp.set_channel_types({'EXG3':'eeg'})           #Mastoid
        rawtemp.set_channel_types({'EXG4':'eeg'})           #Vertex
        rawtemp.set_channel_types({'EXG5':'eeg'})           #Ground
        rawlist += [rawtemp, ]
        evelist += [evestemp, ]
    raw, eves = mne.concatenate_raws(rawlist, events_list=evelist)
    raw, eves = raw.resample(4096, events=eves)
    raw.info['bads'].append('EXG6') 
    raw.info['bads'].append('EXG7')
    raw.info['bads'].append('EXG8')
    raw.info['bads'].append('A26')
    raw.info['bads'].append('A27')
    raw.info['bads'].append('A12')
    
    raw.set_annotations(None)
    
    #raw.set_eeg_reference('average', projection =True)
    
    if subj == 'Q417':
        raw.info['bads'].append('A31')
        raw.info['bads'].append('A32')      
        
    if subj == 'Q414_1':
        raw.info['bads'].append('A1')
        raw.info['bads'].append('EXG5')
        raw.info['bads'].append('EXG4')     
    
    # To check and mark bad channels
    raw.plot(duration=25.0, n_channels=41, scalings=dict(eeg=200e-6))
    
#%% Epoching, evoked, saving 
    raw.filter(1.,20.)
    picks =['A18', 'A2', 'A3', 'A23', 'A8','A22', 'A7','A24','A9']
    
    fs = raw.info['sfreq']
    ## Events 1, 2 for Q414, Q415, Q417
    
    ###Make manual events of 5 seconds for ABR
    # eves_manual = np.zeros(((np.int(raw._data.shape[1])/fs)*5, 3))
    # eves_manual = np.zeros((1350, 3)) ##Calculated using above 
    # eves_manual[:,2] = 2 #Setting event id =2 (as I don't need evoked responses)
    
    # for k in range(1350):
    #     eves_manual[k, 0] = (k + 1)*5*fs
    # eves_manual = np.int64(eves_manual)
    
    epochs = mne.Epochs(raw, eves, event_id=[1], baseline=(-0.2, 0), tmin=-0.2, tmax=1.1, reject=dict(eeg=150e-6), 
                         reject_by_annotation=False, preload=True, proj=False)
    evoked=epochs.average(picks=picks)
    evoked.plot()   

    x = evoked.get_data(picks)    
    t=epochs.times
    
    # plt.plot(t,x.mean(axis=0))
    # plt.show()
    
    mat_ids_ev = dict(evoked=x, picks=picks, fs=4096, t=epochs.times)
    savemat(save_loc_mat + subj + '_1_evoked_anesthetized_1.mat', mat_ids_ev)
    
    
#%% Time-frequency analysis and saving it  
   
    epochs = epochs.copy().subtract_evoked()
    t = epochs.times
    
    theta_freqs = np.arange(4., 8., 0.5) 
    alpha_freqs = np.arange(8., 12., 0.5) 
    beta_freqs = np.arange(12., 35., 0.5)
    low_gamma_freqs = np.arange(35., 55., 1.) 
    high_gamma_freqs= np.arange(55., 80., 1.)
    
    n_cycles1 = theta_freqs*0.4
    n_cycles2 = alpha_freqs*0.4
    n_cycles3 = beta_freqs*0.2
    n_cycles4 = low_gamma_freqs*0.2
    n_cycles5 = high_gamma_freqs*0.2
    

    picks =['A18', 'A2', 'A3', 'A23', 'A8','A22', 'A7','A24','A9']

    theta = tfr_multitaper(epochs, theta_freqs, n_cycles1,time_bandwidth=4, n_jobs=None, return_itc=False,
                           picks=picks)
    alpha = tfr_multitaper(epochs, alpha_freqs, n_cycles2, picks=picks,
                                    time_bandwidth=4, n_jobs=None, return_itc=False)
    
    beta = tfr_multitaper(epochs, beta_freqs, n_cycles3, picks=picks,
                                    time_bandwidth=4, n_jobs=None, return_itc=False)
    low_gamma = tfr_multitaper(epochs, low_gamma_freqs, n_cycles4, picks=picks,
                                    time_bandwidth=4, n_jobs=None, return_itc=False)
    
    high_gamma = tfr_multitaper(epochs, high_gamma_freqs, n_cycles5, picks=picks,
                                    time_bandwidth=4, n_jobs=None, return_itc=False)
    
    beta.plot(baseline=(-0.1, 0.5), mode='mean',title=str (subj+ '-Anesthetized - Beta'),
                     combine='mean')
    alpha.plot(baseline=(-0.1, 0.5), mode='mean',title=str (subj+ '-Anesthetized - Alpha'),
                    combine='mean')

    theta.plot(baseline=(-0.1, 0.5), mode='mean',title=str (subj+ '-Anesthetized -Theta'),
                      combine='mean')
    low_gamma.plot(baseline=(-0.1, 0.5), mode='mean',title=str (subj+ '-Anesthetized - Low Gamma(30-45Hz)'),
                      combine='mean')
    
    high_gamma.plot(baseline=(-0.1, 0.5), mode='mean',title=str (subj+ '-Anesthetized- High Gamma (55-80Hz)'),
                       combine='mean')

    a = theta.data
    b = alpha.data
    c = beta.data
    d = low_gamma.data
    e = high_gamma.data
    
    mat_ids = dict(theta=a,alpha=b, beta=c, low_gamma=d, high_gamma=e, 
                   picks=picks,t=t)
    savemat(save_loc_mat + subj + '_ABR_freqbands_awake.mat', mat_ids)

#%% Plotting Onset responses for all conditions included

    # epochs_down = mne.Epochs(raw, eves, event_id=[1,3], baseline=(-0.3,0), tmin=-0.3, tmax=1.2, reject=dict(eeg=200e-6), 
    #                     reject_by_annotation=False, preload=False, proj=False)
    # all_channels = (np.arange(0,31))
    # picks = (1,2,17,6,22,21)
    # # epoch_12 = epochs_12.get_data()[:, all_channels,:]
    # # ep_mean = epoch_12.mean(axis=1)
    
    # t=epochs_down.times
    # epochs_up = mne.Epochs(raw, eves, event_id=[2,4], baseline=(-0.3,0), tmin=-0.3, tmax=1.2, reject=dict(eeg=200e-6), 
    #                          reject_by_annotation=False, preload=False, proj=False)
    # # epoch_up = epochs_up.get_data()[:,all_channels,:]
    # # epochs_down = epochs_down.get_data()[:,all_channels,:]
    # ep_up_mean = ((epochs_up.get_data()[:,all_channels,:]).mean(axis=1))*1e6
    # ep_down_mean =((epochs_down.get_data()[:,all_channels,:]).mean(axis=1))*1e6
   
    # #plt.plot(t,ep_up_mean.mean(axis=0), label='Up')
    # plt.plot(t,ep_down_mean.mean(axis=0), label='Down')
    # # plt.plot(t,epoch_8, label='8')
    # plt.title(subj + ' - Awake - A8')
    # # plt.xlim(-0.2,0.4)
    # # plt.ylim(-5,15)
    # plt.legend()
    # plt.show()
    
    # plt.savefig(save_loc_fig + subj + '_Awake_5sec_A8', dpi=500)
        
    # mat_ids = dict(ep_up_mean=ep_up_mean, ep_down_mean=ep_down_mean, sfreq=sfreq)
    # io.savemat(save_loc_mat + subj + '_ACC_TrialEpochs.mat', mat_ids)
