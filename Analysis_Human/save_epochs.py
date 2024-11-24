# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 22:47:00 2023

@author: vmysorea
"""
#Saving epochs for full time
import sys
sys.path.append('C:/Users/vmysorea/Documents/mne-python/')
sys.path.append('C:/Users/vmysorea/Documents/ANLffr/')
import warnings
import mne
from anlffr.helper import biosemi2mne as bs
from matplotlib import pyplot as plt
import os
import fnmatch
from anlffr.preproc import find_blinks
from mne import compute_proj_epochs
# import numpy as np
from scipy.io import savemat

plt.switch_backend('QT5Agg')  # Making the plots interactive (Scrollable)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# %% Loading subjects, reading data, mark bad channels
froot = 'D:/PhD/Data/MTB_EP - GDT, Binding, mTRF/Binding/'  # file location
save_loc='C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Human_Analysis/Figures/'
save_epochs_loc = 'D:/PhD/Data/Epochs-fif/'
save_mat_loc = 'D:/PhD/Data/Binding_matfiles/'

subjlist = ['S104']
            
# S271','S281','S290','S284',
               # 'S305','S303','S288','S260', 'S341','S312','S347','S340','S078','S069','S088',
               # 'S072','S308','S344','S105','S345','S291','S310','S339'
# 'S268','S269','S274','S282','S285','S259',
# 'S273','S277','S279','S280','S270',
condlist = [1, 2]  # List of conditions- Coherence of 12 and 20 tones
condnames = ['12', '20']

for subj in subjlist:
    
    fpath = froot + subj + '_Binding/'
    bdfs = fnmatch.filter(os.listdir(fpath), subj +'_Binding*.bdf')      # Load data and read event channel
    print('LOADING! ' + subj +' raw data')

    rawlist = []
    evelist = []

    for k, rawname in enumerate(bdfs):
        rawtemp, evestemp = bs.importbdf(fpath + rawname, verbose='DEBUG', refchans=['EXG1', 'EXG2'])
        rawlist += [rawtemp, ]
        evelist += [evestemp, ]
    raw, eves = mne.concatenate_raws(rawlist, events_list=evelist)
    #raw.plot(duration=25.0, n_channels=32, scalings=dict(eeg=100e-6), event_color={1: 'r', 2: 'g'})    # To check and mark bad channels

#%%Reject a few of the electrodes for each subject  
    if subj == 'S273':
       raw.info['bads'].append('A1')
       raw.info['bads'].append('A30')
       raw.info['bads'].append('A24')
        
    if subj == 'S282':
       raw.info['bads'].append('A16')
       
    if subj == 'S285':
       raw.info['bads'].append('A24')
       
    if subj == 'S280':
       raw.info['bads'].append('A10')
       raw.info['bads'].append('A25')
       
    if subj == 'S259':
       raw.info['bads'].append('A7')
       raw.info['bads'].append('A24')
    
    if subj == 'S270':
       raw.info['bads'].append('A24')
       
    if subj == 'S271':
        raw.info['bads'].append('A24')
        raw.info['bads'].append('A10')
        raw.info['bads'].append('A19')
        raw.info['bads'].append('A17')
   
    if subj == 'S273':
       raw.info['bads'].append('A1')
       raw.info['bads'].append('A30')
       raw.info['bads'].append('A24')
    
    if subj == 'S281':
       raw.info['bads'].append('A29')
       raw.info['bads'].append('A24')
       raw.info['bads'].append('A25')
       raw.info['bads'].append('A28')
       raw.info['bads'].append('A3')
        
    if subj == 'S072':
       raw.info['bads'].append('A1')
       raw.info['bads'].append('A30')
    
    if subj == 'S088':
       raw.info['bads'].append('A6') #not that bad
        
    if subj == 'S303':
       raw.info['bads'].append('A1') #not that bad
       raw.info['bads'].append('A30') #not that bad
       
    if subj == 'S288':
       raw.info['bads'].append('A1') 
       raw.info['bads'].append('A2')
       raw.info['bads'].append('A7') 
       raw.info['bads'].append('A24')
       
    if subj == 'S347':
        raw.info['bads'].append('A16') 
        raw.info['bads'].append('A7')
        raw.info['bads'].append('A28') 
        
    if subj == 'S340':
        raw.info['bads'].append('A1') 
        raw.info['bads'].append('A2')
        raw.info['bads'].append('A7') 
        raw.info['bads'].append('A24')
        raw.info['bads'].append('A30')
        
    if subj == 'S345':
        raw.info['bads'].append('A30') 
        raw.info['bads'].append('A3')
        raw.info['bads'].append('A24') 
        
    if subj == 'S339':
        raw.info['bads'].append('A7') 
        raw.info['bads'].append('A24')

#%% Filtering
    raw.filter(0.4, 40.)
    # raw.info

# %% Blink Rejection
    blinks = find_blinks(raw)
    #raw.plot(events=blinks, duration=25.0, n_channels=32, scalings=dict(eeg=200e-6))
    epochs_blinks = mne.Epochs(raw, blinks, event_id=998, baseline=(-0.25, 0.25), reject=dict(eeg=500e-6), tmin=-0.25, tmax=0.25)
    blink_proj = compute_proj_epochs(epochs_blinks, n_eeg=1)
    raw.add_proj(blink_proj)  # Adding the n=blink proj to the data -- removal of blinks
    #raw.plot_projs_topomap()     # Visualizing the spatial filter

#%% Saving epochs and evoked to fiff files 
    picks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    epochs = mne.Epochs(raw, eves, event_id=[1, 2], proj=True, baseline = (-0.3,0), tmin=-0.3, tmax=5.2, reject=dict(eeg=150e-6),picks=picks)
    # epochs.save(save_epochs_loc + subj + '_Binding_epochs_0.2-epo.fif', fmt = 'double', overwrite=True)
    
    evoked=epochs.average()
    # evoked.save(save_epochs_loc + subj + '_Binding_evoked_0.2-ave.fif', overwrite=True)
    epochs_12 = mne.Epochs(raw, eves, event_id=[1],  proj=True, baseline = (-0.3,0), tmin=-0.3, tmax=5.2, reject=dict(eeg=150e-6), picks=picks)
    # epochs_20.save(save_epochs_loc + subj + '_Binding_epochs20_0.2-epo.fif', fmt = 'double', overwrite=True)
    evoked_12 = epochs_12.average(picks=picks)
    
    epochs_20 = mne.Epochs(raw, eves, event_id=[2],  proj=True, baseline = (-0.3,0), tmin=-0.3, tmax=5.2, reject=dict(eeg=150e-6), picks=picks)
    # epochs_20.save(save_epochs_loc + subj + '_Binding_epochs20_0.2-epo.fif', fmt = 'double', overwrite=True)
    evoked_20 = epochs_20.average(picks=picks)
    # evoked_20.save(save_epochs_loc + subj + '_Binding_evoked20_0.4-ave.fif', overwrite=True)
    
    # epochs_12 = mne.Epochs(raw, eves, event_id=[1], proj=True, tmin=-0.3, tmax=5.2, reject=dict(eeg=150e-6), picks=picks)
    # evoked_12 = epochs_12.average()
    # evoked_12.save(save_epochs_loc + subj + '_Binding_evoked12_0.2-ave.fif', overwrite=True)

#%%Saving epochs and evoked to mat file 
    # a = (epochs.get_data(picks)).dtype=np.int64
    # b = epochs_20.get_data(picks)
    x = evoked.get_data(picks)
    y = evoked_20.get_data(picks)
    # a = evoked.get_data(picks)
    z = evoked_12.get_data(picks)
    # # z = evoked_12.get_data(picks)
    
    t=epochs.times
    # mat_ids_ep = dict(epochs=a, epochs20=b, fs=4096, t=epochs.times)
    mat_ids_ev = dict(evoked=x, evoked20 = y, evoked12=z, fs=4096, t=epochs.times)
    # savemat(save_mat_loc + subj + '_allepochs0.1_NB.mat', mat_ids_ep)
    savemat(save_mat_loc + subj + '_allevoked0.4_with12.mat', mat_ids_ev)
    
    # print('WOOOHOOOO! Saved ' + subj)

    del  epochs_20, evoked_20, evoked_12
