# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 15:13:20 2022
@author: vmysorea
"""
#%% Analysis for 12 and 20 coherence stimuli to test auditory binding 
        # Saves DC shift (300-800 ms) in amplitude for 12 and 20 coh stimuli and calculates coherence - incoherence DC shift 
        # Saves evoked responses for 12 and 20 tone stim for all 32 channels 
        # Has option of saving pickles or mat files 
        
#%% Initializing... 
 
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
import scipy as scipy
from scipy.io import savemat

plt.switch_backend('QT5Agg')  # Making the plots interactive (Scrollable)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# Defining the dimensions and quality of figures
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 120

# %% Loading subjects, reading data, mark bad channels

froot = 'D:/PhD/Data/MTB_EP - GDT, Binding, mTRF/Binding/'  # file location
save_loc='C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Human_Analysis/Figures/'
save_loc_mat= 'D:/PhD/Data/Binding_matfiles/0.4-40Hz/'
save_epochs_loc = 'C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Human_Analysis/Epochs-fif/'

subjlist = ['S104']
 
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
    # raw.plot(duration=25.0, n_channels=32, scalings=dict(eeg=100e-6), event_color={1: 'r', 2: 'g'})    # To check and mark bad channels

#%% Reject a few of the electrodes for each subject
  
    if subj == 'S269':
        raw.info['bads'].append('A25')
        raw.info['bads'].append('A30')
        raw.info['bads'].append('A24')
        
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
        
    if subj == 'S308':
        raw.info['bads'].append('A6') 
        raw.info['bads'].append('A3')
        
    if subj == 'S344':
        raw.info['bads'].append('A24')
        
    if subj == 'S104':
        raw.info['bads'].append('A1')
        raw.info['bads'].append('A6')
        raw.info['bads'].append('A8')
        raw.info['bads'].append('A18')
        raw.info['bads'].append('A19')
        raw.info['bads'].append('A24')
        raw.info['bads'].append('A25')
        raw.info['bads'].append('A17')  
        raw.info['bads'].append('A21')
        raw.info['bads'].append('A29')
        raw.info['bads'].append('A30')
        raw.info['bads'].append('A28')
        
    if subj == 'S337':
        raw.info['bads'].append('A21')
        raw.info['bads'].append('A1')
        raw.info['bads'].append('A30')
        raw.info['bads'].append('A2')
        raw.info['bads'].append('A29')
    
    if subj == 'S308':
       raw.info['bads'].append('A7') 
       raw.info['bads'].append('A11') 
       raw.info['bads'].append('A24') 
       
#%% Filtering for cortical responses 

    raw.filter(1, 40.)
    raw.info

# %% Blink Rejection

    blinks = find_blinks(raw)
    raw.plot(events=blinks, duration=25.0, n_channels=32, scalings=dict(eeg=200e-6))
    epochs_blinks = mne.Epochs(raw, blinks, event_id=998, baseline=(-0.25, 0.25), reject=dict(eeg=500e-6), tmin=-0.25, tmax=0.25)
    blink_proj = compute_proj_epochs(epochs_blinks, n_eeg=1)
    raw.add_proj(blink_proj)  # Adding the n=blink proj to the data -- removal of blinks
    raw.plot_projs_topomap()     # Visualizing the spatial filter

# %% Plotting Onset responses

#Plotting onsets for each event
# for subj in subjlist:
#     for c, cond in enumerate(condlist):
#         condname = condnames[c]
#         epochs = mne.Epochs(raw, eves, cond, tmin=-0.3, proj=True, tmax=1.1, baseline=(-0.3, 0.0),reject=dict(eeg=200e-6))
#         evoked = epochs.average()
#         evokeds += [evoked, ]
#         evoked.plot(titles=subj + 'Onset Response -' + condname) #picks=picks

#Plotting onsets for both events combined
    epochs = mne.Epochs(raw, eves, event_id=[1, 2], baseline=(-0.3, 0), proj=True, tmin=-0.3, tmax=1.1, reject=dict(eeg=200e-6))
    evoked = epochs.average()
    Onset_Total = evoked.plot(titles= subj + 'Combined Onset - 12,20')
#OnsetResponse_All3.savefig(save_loc + 'OnsetResponse_All3_.png' + subj, dpi=300)

# ##Plotting full time
# epochs = mne.Epochs(raw, eves, event_id=[1, 2], baseline=(-0.3, 0), proj=True, tmin=-0.3, tmax=5.2, reject=dict(eeg=150e-6))
# evoked = epochs.average()
# highlight = [(0,0.2), (0.3,0.8), (1.0,1.2), (1.3,1.8), (2.0,2.2), (2.3,2.8),(3.0,3.2), (3.3,3.8), (4.0,4.2), (4.3,4.8)]
# Onset_ftime = evoked.plot(spatial_colors=True,  gfp=True, picks='A32')   #Butterfly plots

# # times = np.arange(-0.2,5.1, 0.3)
# times=(0.2, 0.55,1.15,1.55,2.2,2.55,3.1,3.55,4.2,4.55)
# evoked.plot_topomap(times=times, average=0.4, cmap='Spectral_r', contours=10, ncols=4, nrows='auto', res=32, colorbar=True)
# #gc = mne.grand_average(evokeds)
# evoked.plot_joint(times, picks = ['A32', 'A31', 'A5', 'A26'])

#mne.viz.plot_evoked_topomap(gc, times='peaks', average=0.4, cmap='Spectral_r', contours=10, ncols=4, nrows='auto', res=32, colorbar=True)
#%% Add events for AB transitions at t = 1,2,3,4

    eves_AB = eves.copy()
    fs = raw.info['sfreq']
    
    for cond in range(2):
        for e in range(4):
            evnt_num = 3 + e + cond*4
            events_add = eves[eves[:,2] == int(cond+1),:] + [int(fs*(e+1)),int(0),evnt_num - (cond+1)]
            eves_AB = np.concatenate((eves_AB,events_add),axis=0)

#%% Extract Different Conditions

    conds = ['12_0', '20_0', '12_AB1', '12_BA1', '12_AB2', '12_BA2', '20_AB1','20_BA1','20_AB2','20_BA2']
    reject=dict(eeg=150e-6)
    
    epochs = []
    evkd = []
    for cnd in range(10):
        ep_cnd = mne.Epochs(raw,eves_AB,cnd+1,tmin=-0.2,tmax=1.1, reject = reject, baseline = (-0.1,0.))
        epochs.append(ep_cnd)
        evkd.append(ep_cnd.average())
        #evkd[cnd].plot(picks=31,titles=conds[cnd])
    
    conds.extend(['12AB', '12BA','20AB','20BA'])
    ev_combos = [[2,4],[3,5],[6,8],[7,9]]
    
    for it, cnd in enumerate(range(10,14)):
        ep_cnd = mne.Epochs(raw,eves_AB,list(np.array(ev_combos[it])+1),tmin=-0.2,tmax=1.1, reject = reject, baseline = (-0.1,0.))
        epochs.append(ep_cnd)
        evkd.append(ep_cnd.average())
        #evkd[cnd].plot(picks=31,titles=conds[cnd])
    
    # Also get whole interval without baselining each interval
    conds.extend(['12', '20'])
    ep_cnd = mne.Epochs(raw,eves,1,tmin=-0.3,tmax=5.5, reject = reject, baseline = (-0.1,0.))
    epochs.append(ep_cnd)
    evkd.append(ep_cnd.average())
    
    ep_cnd = mne.Epochs(raw,eves,2,tmin=-0.3,tmax=5.5, reject = reject, baseline = (-0.1,0.))
    epochs.append(ep_cnd)
    evkd.append(ep_cnd.average())

#%% Plot 1st and second interval

# for it, c in enumerate(ev_combos):
#     evkds = [evkd[c[0]], evkd[c[1]]]
#     #mne.viz.plot_compare_evokeds(evkds,picks=31,title = conds[it + 10])
#%% Plot Comparisons

# combos_comp = [[0,1], [10,12], [11,13]]
# comp_labels = ['Onset', 'Incoherent to Coherent', 'Coherent to Incoherent']

# for it,c in enumerate(combos_comp):
#     evkds = [evkd[c[0]], evkd[c[1]]]
#     #mne.viz.plot_compare_evokeds(evkds,title=comp_labels[it]
#%% Make Plots outside of MNE
    picks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    combos_comp = [[0,1], [10,12], [11,13]]
    comp_labels = ['Onset', 'Incoherent to Coherent', 'Coherent to Incoherent']
    
    fig, ax = plt.subplots(3,1,sharex=True)
    
    t = epochs[0].times
    
    for cnd in range(len(combos_comp)):
        cz_12 = (epochs[combos_comp[cnd][0]].get_data()[:,picks,:]).mean(axis=1)
        cz_mean_12 = cz_12.mean(axis=0)
        cz_sem_12 = scipy.stats.sem(cz_12, axis=0)
    
        cz_20 = (epochs[combos_comp[cnd][1]].get_data()[:,picks,:]).mean(axis=1)
        cz_mean_20 = cz_20.mean(axis=0)
        cz_sem_20 = scipy.stats.sem(cz_20, axis=0)
        #cz_sem_20 = cz_ep_20.std(axis=0) / np.sqrt(cz_ep_20.shape[0])
    
        ax[cnd].plot(t,cz_mean_12,label='12')
        ax[cnd].fill_between(t,cz_mean_12 - cz_sem_12, cz_mean_12 + cz_sem_12,alpha=0.5)
    
        ax[cnd].plot(t,cz_mean_20,label='20')
        ax[cnd].fill_between(t,cz_mean_20 - cz_sem_20, cz_mean_20 + cz_sem_20,alpha=0.5)
    
        ax[cnd].set_title(comp_labels[cnd])
        ax[cnd].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    
    ax[0].legend()
    ax[2].set_xlabel('Time (sec)')
    ax[1].set_ylabel('Amplitude (' + u"\u03bcA" + ')')
    #ax.set_ylim (-5 * 1e-6 , 5 * 1e-6)
    plt.suptitle(subj + '_Binding')
    plt.show()
#plt.savefig(os.path.join(save_loc,subj + '_12vs20.png'),format='png', dpi=300)

#%% Calculate avg shift from baseline during A-B and B-A from 300-800 ms

    combos_comp = [[0,1], [10,12], [11,13]]
    comp_labels = ['Onset', 'Incoherent to Coherent', 'Coherent to Incoherent']

    t1 = t>=0.3
    t2 = t<=0.8
    t3 = np.array([t2[i] and t1[i] for i in range(len(t1))])
    #t3= np.logical_and(t1,t2) #Subtracting t2-t1
    cz_12_t3_all=[]
    cz_20_t3_all=[]

    for cnd in range(len(combos_comp)):
        cz_12 = (epochs[combos_comp[cnd][0]].get_data()[:,picks,:]).mean(axis=1)
        cz_12_t3_Onset = cz_12[:,t3]
        cz_12_t3_avg = abs(cz_12_t3_Onset).mean(axis=0)
        cz_12_t3_all += (cz_12_t3_avg,)
        print(subj + '-12 Shift from baseline=' , cz_12_t3_avg)

        cz_20 = (epochs[combos_comp[cnd][1]].get_data()[:,picks,:]).mean(axis=1)
        cz_20_t3_Onset = cz_20[:,t3]
        cz_20_t3_avg = cz_20_t3_Onset.mean(axis=0)
        cz_20_t3_all += (cz_20_t3_avg,)
        print(subj + '-20 Shift from baseline=' , cz_20_t3_avg)
        
    #%% Subtracting *coherent-incoherent* baseline shift to be similar to GFP for mean of 32 channels 
    
    dcshift_12 = cz_12_t3_all [combos_comp [1][0] - combos_comp [2][0]]
    dcshift_20 = cz_20_t3_all [combos_comp [1][1] - combos_comp [2][1]]
    
    mat_ids = dict(combos_comp = combos_comp, cz_12_t3_all=cz_12_t3_all ,cz_20_t3_all=cz_20_t3_all, 
                   dcshift_12=dcshift_12, dcshift_20=dcshift_20)
    savemat(save_loc_mat + subj + '_Binding(300-800ms)_1-40_Shifts_AllChans.mat', mat_ids)

#%% Save Epochs, evokeds for 32 channels for 8 different conditions - Pickles and mat files 

    save_indexes = [0,1,10,11,12,13,14,15]      # 0 - 12 Onset (Upto 1.1 s)
                                                # 1 - 20 Onset (Upto 1.1 s)
                                                # 10 - 12 Incoherent to Coherent
                                                # 11 - 20 Incoherent to Coherent
                                                # 12 - 12 Coherent to Incoherent 
                                                # 13 - 20 Coherent to Incoherent 
                                                # 14 - 12 Full 5 seconds
                                                # 15 - 20 Full 5 seconds 
    conds_save = []
    epochs_save = []
    evkd_save = [] #save 32 channel evkd response
    t = t 
    t_full = epochs[-1].times

    for si in save_indexes:
          conds_save.append(conds[si])
          evkd_save.append(evkd[si])
          # epochs_save.append((epochs[si].get_data()[:,picks,:]).mean(axis=1)) # Saving epochs 

    # pickle_loc='D:/PhD/Data/Binding_Pickles/'

    # with open(os.path.join(pickle_loc,subj+'_Binding_0.4_AllChan.pickle'),'wb') as file:
    #     pickle.dump([t, t_full, conds_save, epochs_save,evkd_save],file)
    mat_ids1 = dict(save_indexes=save_indexes, conds_save = conds_save, 
                        evkd0 =evkd_save[0].data, 
                        evkd1 =evkd_save[1].data, 
                        evkd2 =evkd_save[2].data, 
                        evkd3 =evkd_save[3].data, 
                        evkd4 =evkd_save[4].data, 
                        evkd5 =evkd_save[5].data, 
                        evkd6 =evkd_save[6].data, 
                        evkd7 =evkd_save[7].data, 
                        t=t, t_full=t_full)
    savemat(save_loc_mat + subj + '_1-40Hz_Evoked_AllChan.mat', mat_ids1)
    
    print ('Woohoooo! Saved -- ' + str(subj) + '!!')
    
del epochs, evkd, evkd_save,epochs_save, cz_20, cz_12