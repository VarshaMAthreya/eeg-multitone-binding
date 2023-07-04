# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 21:25:43 2023

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
#from mne.time_frequency import tfr_multitaper
from scipy.io import savemat
import pickle

plt.switch_backend('QT5Agg')  # Making the plots interactive (Scrollable)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# Defining the dimensions and quality of figures
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 120

#%% Setting locs and loading data

froot = 'D:/PhD/Data/Chin_Data/Awake/'  # file location
save_loc='C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Chin_Analysis/Figures/'
pickle_loc = 'C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Chin_Analysis/Chin_Pickles/'

subjlist = ['Q414_2']  # Load subject folder
condlist = [1] 
condnames = ['12']

for subj in subjlist:
    evokeds = []
    # Load data and read event channel
    fpath = froot + subj + '/'
    bdfs = fnmatch.filter(os.listdir(fpath), subj +
                          '_Awake_Binding*.bdf')

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
    
    # if subj == 'Q414_1':
    #     raw.set_channel_types({'EXG2':'eeg'})           #Mastoid - 35
    #     raw.set_channel_types({'EXG3':'eeg'})           #Vertex - 34
    #     raw.set_channel_types({'EXG1':'eeg'})           #Ground - 33
        
    #raw.set_eeg_reference(ref_channels='average')

    # To check and mark bad channels
    raw.plot(duration=25.0, n_channels=41, scalings=dict(eeg=200e-6))
    
# %% Filtering
    raw.filter(1., 20.)
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

#%%Get subdermal data and plot 
    t=epochs[0].times
    all_channels = (np.arange(1,32))
    ep_mastoid = epochs.get_data()[:,34,:] #Mastoid -EXG3
    ep_vertex= epochs.get_data()[:,35,:] #Vertex -EXG4
    ep_ground = epochs.get_data()[:,36,:] #Ground - EXG5
    ep_all = evoked.data[all_channels,:]
    ep_mean_all =ep_all.mean(axis=0)
    ep_sem_all = ep_all.std(axis=0) / np.sqrt(ep_all.shape[0])
    
    ep_subderm = ep_vertex - ep_mastoid #Inverting mastoid and non-inverting vertex 
    ep_mean_subderm = ep_subderm.mean(axis=0)
    ep_sem_subderm = ep_subderm.std(axis=0) / np.sqrt(ep_subderm.shape[0])
    
    plt.plot(t, ep_mean_subderm, label='Subdermal electrode')
    plt.fill_between(t, ep_mean_subderm - ep_sem_subderm,
                         ep_mean_subderm + ep_sem_subderm,alpha=0.5)
    plt.plot(t, ep_mean_all, label = 'EEG Cap')
    plt.fill_between(t, ep_mean_all - ep_sem_all,
                          ep_mean_all + ep_sem_all,alpha=0.5)
    #plt.xticks(ticks= [-2, 0, 2, 4, 6, 8, 10, 12, 14])
    plt.title('Onsets - Awake Binding')
    plt.legend()
    plt.show()

#%% Add events for AB transitions at t = 1,2,3,4
    eves_AB = eves.copy()
    fs = raw.info['sfreq']
    
    for cond in range(1):
        for e in range(4):                   # 4 transition points t = 1,2,3,4
            evnt_num = 2 + e + cond*4       #Changed 3 to 2, as there is no event id=2 in the original dataset (Only 1 condition)
            events_add = eves[eves[:,2] == int(cond+1),:] + [int(fs*(e+1)),int(0),evnt_num - (cond+1)]
            eves_AB = np.concatenate((eves_AB,events_add),axis=0)

#%%Extracting different conditions
    conds = ['12_0','12_AB1','12_BA1','12_AB2','12_BA2']
    reject=dict(eeg=150e-6)
    
    epochs = []
    evkd = []
    for cnd in range(5):
        ep_cnd = mne.Epochs(raw,eves_AB,cnd+1,tmin=-0.3,tmax=1.2, reject = reject, baseline = (-0.3,0.))
        epochs.append(ep_cnd)
        evkd.append(ep_cnd.average())
        #evkd[cnd].plot(picks=31,titles=conds[cnd])
    
    conds.extend(['12AB','12BA'])
    ev_combos=[[1,3],[2,4]]
    
    for it, cnd in enumerate(range(5,7)):
        ep_cnd = mne.Epochs(raw,eves_AB,list(np.array(ev_combos[it])),
                            tmin=-0.3,tmax=1.2,
                            reject = reject, baseline = (-0.3,0.))
        epochs.append(ep_cnd)
        evkd.append(ep_cnd.average())
        #evoked[cnd].plot(picks=picks,titles=conds[cnd])
    
    # Also get whole interval without baselining each interval
    conds.extend(['12'])
    ep_cnd = mne.Epochs(raw,eves,event_id=[1],tmin=-0.3,tmax=5.5, reject = reject, baseline = (-0.3,0.))
    epochs.append(ep_cnd)
    evkd.append(ep_cnd.average())
    #evoked.plot()

#%% Plotting cap and subdermal responses
    combos_comp = [0, 5, 6]
    comp_labels = ['Onset', 'Incoherent to Coherent', 'Coherent to Incoherent']
    
    t=epochs[0].times
    channels = [7]
    
    fig, ax = plt.subplots(3,1,sharex=True)
    for cnd in range(len(combos_comp)):
        for chan in range(len(channels)):
            cz_ep_12 = epochs[combos_comp[cnd]].get_data()[:,chan,:]
            cz_mean_12 = cz_ep_12.mean(axis=0)
            cz_sem_12 = cz_ep_12.std(axis=0) / np.sqrt(cz_ep_12.shape[0])            
       
            ep_mastoid_12 = epochs[combos_comp[cnd]].get_data()[:,34,:]                      #Mastoid -EXG3
            ep_vertex_12 = epochs[combos_comp[cnd]].get_data()[:,35,:]                       #Vertex -EXG4
            # ep_ground_12 = epochs[combos_comp[cnd][1]].get_data()[:,37,:]                       #Ground - EXG
            ep_subderm_12 = ep_vertex_12 - ep_mastoid_12 #Inverting mastoid and non-inverting vertex 
            ep_mean_subderm_12 = ep_subderm_12.mean(axis=0)
            ep_sem_subderm_12 = ep_subderm_12.std(axis=0) / np.sqrt(ep_subderm_12.shape[0])
        
            ax[cnd].plot(t,cz_mean_12,label='12_EEG Cap')
            ax[cnd].fill_between(t,cz_mean_12 - cz_sem_12, cz_mean_12 + cz_sem_12,alpha=0.5)
            
            ax[cnd].plot(t, ep_mean_subderm_12,label='12_Subdermal')
            ax[cnd].fill_between(t,ep_mean_subderm_12 -  ep_sem_subderm_12, ep_mean_subderm_12 +  ep_sem_subderm_12,alpha=0.5)
        
            ax[cnd].set_title(comp_labels[cnd])
            ax[cnd].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
            # ax[cnd].set_ylim([-4*1e-6, 4*1e-6])
            # ax.set_xlim([-0.2, 1.1])
        
    ax[0].legend(prop={'size': 6})
    ax[2].set_xlabel('Time(sec)')
    ax[1].set_ylabel('Amplitude (\u03bcV)')
    plt.suptitle(subj + '_Binding - Awake')
    plt.show()

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