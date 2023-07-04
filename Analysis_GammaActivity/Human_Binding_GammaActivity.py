# -*- coding: utf-8 -*-
"""
Created on Tue May  2 01:23:51 2023

@author: vmysorea
"""
import sys
sys.path.append('C:/Users/vmysorea/Documents/mne-python/')
sys.path.append('C:/Users/vmysorea/Documents/ANLffr/')
import warnings
import mne
import numpy as np
from anlffr.helper import biosemi2mne as bs
from matplotlib import pyplot as plt
import os
import fnmatch
from anlffr.preproc import find_blinks
from mne import compute_proj_epochs
# import pickle
# import scipy as scipy
from scipy.io import savemat
# from mne.stats import bootstrap_confidence_interval
# from mne.baseline import rescale
from mne.time_frequency import tfr_multitaper

plt.switch_backend('QT5Agg')  # Making the plots interactive (Scrollable)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# Defining the dimensions and quality of figures
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 120


#%% Setting locs and loading data

froot = 'D:/PhD/Data/MTB_EP - GDT, Binding, mTRF/Binding/'   # file location
save_loc_mat='C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Human_Analysis/Matfiles(Avg)/Gamma_Matfiles/'

#%%Evoked Response -- Get EFR 
   
subjlist =['S310'] 
# subjlist = ['S273','S268','S269', 'S274', 'S282', 'S285',
#             'S277', 'S279', 'S280','S259','S270', 'S271', 
#             'S281','S290', 'S284', 'S305','S303','S288','S260',
#             'S352', 'S341', 'S312', 'S347', 'S340','S078','S069',
#             'S088','S342','S072','S308','S344','S345','S291','S310','S339'] # Load subject folder

condlist = [1, 2]  # List of conditions- Coherence of 12 and 20 tones
condnames = ['12', '20']

evokeds = []

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
    # To check and mark bad channels
    # raw.plot(duration=25.0, n_channels=41, scalings=dict(eeg=200e-6))

#%% Reject a few of the electrodes for each subject
  
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
    
    if subj == 'S284':
       raw.info['bads'].append('A3')
       raw.info['bads'].append('A7')
       
    if subj == 'S259':
       raw.info['bads'].append('A7')
       raw.info['bads'].append('A24')
    
    if subj == 'S270':
       raw.info['bads'].append('A24')
       
    if subj == 'S271':
        raw.info['bads'].append('A24')
        raw.info['bads'].append('A10')
   
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
        raw.info['bads'].append('A6')
        
    if subj == 'S339':
        raw.info['bads'].append('A7') 
        raw.info['bads'].append('A24')
        
    if subj == 'S105':
        raw.info['bads'].append('A24') 
        raw.info['bads'].append('A25')
        
    if subj == 'S342':
        raw.info['bads'].append('A29') 
            
#%% Filtering
    raw.filter(1., 90.,)
    raw.info
    
    raw.notch_filter(np.arange(60, 241, 60), picks='A32', filter_length='auto',
                 phase='zero')

# %% Blink Rejection
    blinks = find_blinks(raw)
    # raw.plot(events=blinks, duration=25.0, n_channels=32, scalings=dict(eeg=200e-6))
    epochs_blinks = mne.Epochs(raw, blinks, event_id=998, baseline=(-0.25, 0.25), reject=dict(eeg=500e-6), tmin=-0.25, tmax=0.25)
    blink_proj = compute_proj_epochs(epochs_blinks, n_eeg=1)
    raw.add_proj(blink_proj)  # Adding the n=blink proj to the data -- removal of blinks
    # raw.plot_projs_topomap()     # Visualizing the spatial filter
    
#%% Plotting Onset responses for all conditions included

    epochs_12 = mne.Epochs(raw, eves, event_id=[1], baseline=(-0.3,0), tmin=-0.3, tmax=5.1, reject=dict(eeg=200e-6), 
                        reject_by_annotation=True, preload=True, proj=True)
    # all_channels = (np.arange(0,31))
    # picks = [8, 21, 11, 12, 18, 31]
    
    t=epochs_12.times
    epochs_20 = mne.Epochs(raw, eves, event_id=[2], baseline=(-0.3,0), tmin=-0.3, tmax=5.1, reject=dict(eeg=200e-6), 
                              reject_by_annotation=True, preload=True, proj=True)

    # ep_12 = ((epochs_12.get_data()[:,picks,:]).mean(axis=1))*1e6
    ep20 = epochs_20.get_data()[:,31,:]
    ep_20 = ep20.mean(axis=1)
   
    # plt.plot(t,ep_12.mean(axis=0), label='12')
    # plt.plot(t,ep_20.mean(axis=0), label='20')
    # plt.title(subj + ' - Binding Gamma- A9, A22, A12, A13, A19, A32')
    # # plt.xlim(-0.2,0.4)
    # # plt.ylim(-5,15)
    # plt.legend()
    # plt.show()
    
    # plt.savefig(save_loc_fig + subj + '_Awake_5sec_A8', dpi=500)
        
    # mat_ids = dict(ep_up_mean=ep_up_mean, ep_down_mean=ep_down_mean, sfreq=sfreq)
    # io.savemat(save_loc_mat + subj + '_ACC_TrialEpochs.mat', mat_ids)
#%% Trial 2
    _, ax = plt.subplots()
    spectrum = epochs_20.compute_psd(fmin=30, fmax=45, tmax=5.1, n_jobs=None, picks=['A32','A31','A5','A26'])
    # average across epochs first
    mean_spectrum = spectrum.average()
    psds, freqs = mean_spectrum.get_data(return_freqs=True)
    # then convert to dB and take mean & standard deviation across channels
    psds = 10 * np.log10(psds)
    psds_mean = psds.mean(axis=0)
    psds_std = psds.std(axis=0)
    
    ax.plot(freqs, psds_mean, color='k')
    ax.fill_between(freqs, psds_mean - psds_std, psds_mean + psds_std,
                    color='k', alpha=.5, edgecolor='none')
    ax.set(title='Multitaper PSD - 20 (A32,A31,A5,A26)', xlabel='Frequency (Hz)',
           ylabel='Power Spectral Density (dB)')

#%% Looking at gamma bands 
    freqs = np.arange(30.,80.,2.)
    n_cycles=freqs*0.4
    
    low_freqs = np.arange(30., 45., 1.)
    high_freqs= np.arange(55., 80., 1.)
    n_cycles1 = low_freqs*0.4
    n_cycles2 = high_freqs*0.4
    
    picks = ['A32']
    epochs12 = epochs_12.copy().subtract_evoked()
    epochs20 = epochs_20.copy().subtract_evoked()
    
    t = epochs12.times
    
    power12_all = tfr_multitaper(epochs12, freqs, n_cycles, picks=picks,
                                    time_bandwidth=4, n_jobs=-1, return_itc=False)
    power20_all = tfr_multitaper(epochs20, freqs, n_cycles, picks=picks,
                                    time_bandwidth=4, n_jobs=-1, return_itc=False)
    
    power12_low = tfr_multitaper(epochs12, low_freqs, n_cycles1, picks=picks,
                                    time_bandwidth=4, n_jobs=-1, return_itc=False)
    power20_low = tfr_multitaper(epochs20, low_freqs, n_cycles1, picks=picks,
                                    time_bandwidth=4, n_jobs=-1, return_itc=False)
    
    power12_high = tfr_multitaper(epochs12, high_freqs, n_cycles2, picks=picks,
                                    time_bandwidth=4, n_jobs=-1, return_itc=False)
    power20_high = tfr_multitaper(epochs20, high_freqs, n_cycles2, picks=picks,
                                    time_bandwidth=4, n_jobs=-1, return_itc=False)
    
 
    power12_all.plot([0], baseline=(-0.3, 0), mode='mean',title=str (subj+ '-Power 12- Gamma (30-80Hz)'),
                     vmin=-3*1e-10, vmax=3*1e-10)
    power20_all.plot([0], baseline=(-0.3, 0), mode='mean',title=str (subj+ '-Power 20- Gamma(30-80Hz)'),
                    vmin=-3, vmax=3)
    
    power12_low.plot([0], baseline=(-0.3, 0), mode='mean',title=str (subj+ '-Power 12- Low Gamma (30-45Hz)'),
                     vmin=-3, vmax=3)
    power20_low.plot([0], baseline=(-0.3, 0), mode='mean',title=str (subj+ '-Power 20- Low Gamma(30-45Hz)'),
                     vmin=-3, vmax=3)

    power12_high.plot([0], baseline=(-0.3, 0), mode='mean',title=str (subj+ '-Power 12- High Gamma (55-80Hz)'),
                      vmin=-3, vmax=3)
    power20_high.plot([0], baseline=(-0.3, 0), mode='mean',title=str (subj+ '-Power 20- High Gamma(55-80Hz)'),
                      vmin=-3, vmax=3)
     
    a = power12_all.data
    b = power20_all.data
    c = power12_low.data
    d = power20_low.data
    e = power12_high.data
    f = power20_high.data
    
    mat_ids = dict(power12_all=a, power20_all=b,power12_low=c, power20_low=d,
                   power12_high=e, power20_high=f, freqs=freqs, low_freqs=low_freqs,
                   high_freqs=high_freqs,picks=picks, n_cycles=n_cycles, n_cycles1=n_cycles1,
                   n_cycles2=n_cycles2, t=t)
    savemat(save_loc_mat + subj + '_A32_gamma_1.mat', mat_ids)
    
    del epochs_12, epochs_20, epochs12, epochs20, a, b, c, d, e, f, n_cycles, n_cycles1, n_cycles2

