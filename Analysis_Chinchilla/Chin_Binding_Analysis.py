# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 18:28:25 2022

@author: vmysorea
"""
#%%Chinchilla Binding Analysis -- Incorporating subdermal electrodes as well 

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
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 120

#%% Setting locs and loading data

froot = 'D:/PhD/Data/Chin_Data/Anesthetized/'  # file location
save_loc='C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Chin_Analysis/Figures/'
pickle_loc = 'C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Chin_Analysis/Chin_Pickles/'

subjlist = ['Q414']  # Load subject folder
condlist = [1, 2] #Coherence of 12 and 8 tones (Reversed here!!!!!!)
condnames = ['12', '8']

for subj in subjlist:
    evokeds = []
    # Load data and read event channel
    fpath = froot #+ subj + '/'
    bdfs = fnmatch.filter(os.listdir(fpath), subj +
                          '_Anesthetized_Binding*.bdf')

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
    
    #raw.set_eeg_reference('average', projection =True)

    # To check and mark bad channels
    raw.plot(duration=25.0, n_channels=41, scalings=dict(eeg=200e-6))

# %% Filtering
raw.filter(1., 40.)
raw.info
raw.plot(duration=25.0, n_channels=41, scalings=dict(eeg=200e-6))

#%% Plotting Onset responses for all conditions included
#picks = ['A8']
for c, cond in enumerate(condlist):
    condname = condnames[c]
    epochs = mne.Epochs(raw, eves, cond, tmin=-0.3, proj=True,
        tmax=1.1, baseline=(-0.3, 0.0),
        reject=dict(eeg=150e-6))
    evoked = epochs.average()
    evokeds += [evoked, ]
    #picks = ['EXG3']
   #evoked.plot(titles='Onset Response - Q414 (Awake)')  #Add picks=picks to see one channel's response

epochs = mne.Epochs(raw, eves, event_id=[1, 2], baseline=(-0.3, 0), proj=True,
                    tmin=-0.3, tmax=1.2, reject=dict(eeg=150e-6))#, picks=picks)
evoked = epochs.average()
# evoked.plot_topomap(0.3)
#picks = ['A8']
OnsetResponse_Total = evoked.plot (spatial_colors=True) #(picks=picks)    #Butterfly plots
#OnsetResponse_AllTotal.savefig(save_loc + 'OnsetResponse_All3_.png' + subj, dpi=300)

#Looking at full time 
epochs = mne.Epochs(raw, eves, event_id=[1, 2], baseline=(-0.3, 0), proj=True,
                    tmin=-0.3, tmax=1.2, reject=dict(eeg=150e-6))#, picks=picks)
evoked = epochs.average()
#evoked.plot_topomap(0.3)
#picks = ['A8']
OnsetResponse_Total = evoked.plot (spatial_colors=True) #(picks=picks)    #Butterfly plots
#OnsetResponse_AllTotal.savefig(save_loc + 'OnsetResponse_All3_.png' + subj, dpi=300)
#%% Plot individual onsets for 8 and 12 coherence stim
epochs_12 = mne.Epochs(raw, eves, event_id=[1], baseline=(-0.3, 0), proj=True,
                    tmin=-0.3, tmax=1.2, reject=dict(eeg=150e-6))#, picks=picks)
evoked_12 = epochs_12.average()
#evoked.plot_topomap(0.3)
#picks = ['A8']
OnsetResponse_12tone = evoked_12.plot(titles='12tone coherence',spatial_colors=True) #, picks=picks)
#evoked_12.plot_topomap()

epochs_8 = mne.Epochs(raw, eves, event_id=[2], baseline=(-0.3, 0),
                    tmin=-0.3, tmax=1.2, reject=dict(eeg=150e-6))#, picks=picks)
evoked_8 = epochs_8.average()
#evoked.plot_topomap(0.3)an
#picks = ['A8']
OnsetResponse_8tone = evoked_8.plot(titles='8tone coherence',spatial_colors=True)
#evoked_8.plot_topomap()

evokeds = dict(coh_12=list(epochs['1'].iter_evoked()),
               coh_8=list(epochs['2'].iter_evoked()))
mne.viz.plot_compare_evokeds(evokeds, combine='mean', title = 'Q414-Awake')

#plt.savefig(os.path.join(save_loc,subj + '_Onset12vs20_Cap.png'),format='png', dpi=300)

#%%Get subdermal data and plot onsets
t=evoked.times
all_cap_channels = (np.arange(1,33))

#8 tone coherence
ep_mastoid_8 = epochs_8.get_data()[:,35,:] #Mastoid -EXG3
ep_vertex_8 = epochs_8.get_data()[:,36,:] #Vertex -EXG4
# ep_ground_8 = epochs_8.get_data()[:,37,:] #Ground - EXG5
ep_all_8 = evoked_8.data[all_cap_channels,:]
ep_mean_all_8 =ep_all_8.mean(axis=0)
ep_sem_all_8 = ep_all_8.std(axis=0) / np.sqrt(ep_all_8.shape[0])

ep_subderm_8 = ep_vertex_8 - ep_mastoid_8 #Inverting mastoid and non-inverting vertex 
ep_mean_subderm_8 = ep_subderm_8.mean(axis=0)
ep_sem_subderm_8 = ep_subderm_8.std(axis=0) / np.sqrt(ep_subderm_8.shape[0])

#12 tone coherence
ep_mastoid_12 = epochs_12.get_data()[:,35,:] #Mastoid -EXG3
ep_vertex_12 = epochs_12.get_data()[:,36,:] #Vertex -EXG4
# ep_ground_12 = epochs_12.get_data()[:,37,:] #Ground - EXG5
ep_all_12 = evoked_12.data[all_cap_channels,:]
ep_mean_all_12 =ep_all_12.mean(axis=0)
ep_sem_all_12 = ep_all_12.std(axis=0) / np.sqrt(ep_all_12.shape[0])

ep_subderm_12 = ep_vertex_12 - ep_mastoid_12 #Inverting mastoid and non-inverting vertex 
ep_mean_subderm_12 = ep_subderm_12.mean(axis=0)
ep_sem_subderm_12 = ep_subderm_12.std(axis=0) / np.sqrt(ep_subderm_12.shape[0])

#Plot onsets
fig, ax = plt.subplots(2,1,sharex=True,sharey=(True),constrained_layout=True)

ax[0].errorbar(t, ep_mean_subderm_8, yerr=ep_sem_subderm_8,
               color='green', linewidth=2, ecolor='palegreen', label ='Subdermal Electrodes')
ax[0].errorbar(t, ep_mean_all_8, yerr=ep_sem_all_8,
               color='purple', linewidth=2, ecolor='thistle', label='EEG Cap - 32 Electrodes')
ax[1].errorbar(t, ep_mean_subderm_12, yerr=ep_sem_subderm_12,
               color='green', linewidth=2, ecolor='palegreen')
ax[1].errorbar(t, ep_mean_all_12, yerr=ep_sem_all_12,
               color='purple', linewidth=2, ecolor='thistle')
ax[0].set_title('8 tone coherence', loc='center', fontsize=10)
ax[1].set_title('12 tone coherence', loc='center', fontsize=10)
ax[0].legend(prop={'size': 6})
#plt.xlim([-0.2, 1.1])
plt.ylim([-1*1e-5,1*1e-5])
plt.xlabel('Time (in seconds)')
fig.text(0.0001, 0.5, 'Amplitude (\u03bcV)', va='center', rotation='vertical')
#ax[1].set_ylabel('Amplitude (\u03bcV)')
plt.suptitle('Onsets - Awake Chinchilla')
plt.rcParams["figure.figsize"] = (5.5, 5)
plt.show()

#plt.savefig(save_loc + 'Onset_12_SB_Cap.png', dpi=300)

#%% Add events for AB transitions at t = 1,2,3,4
eves_AB = eves.copy()
fs = raw.info['sfreq']

for cond in range(2):               # 2 conditions (8, 12)
    for e in range(4):              # 4 transition points t = 1,2,3,4
        event_num = 3 + e + cond*4
        events_add = eves[eves[:,2] == int(cond+1),:] + [int(fs*(e+1)),int(0),event_num - (cond+1)]
        eves_AB = np.concatenate((eves_AB,events_add),axis=0)

#%% Extract Different Conditions
###### The OG
conds = ['12_0', '8_0', '12_AB1','12_BA1','12_AB2','12_BA2', '8_AB1', '8_BA1', '8_AB2', '8_BA2']
reject = dict(eeg=150e-6)

epochs = []
evoked = []
#picks = ['A3']
for cond in range(10):
    ep_cond = mne.Epochs(raw,eves_AB,cond+1,tmin=-0.3,tmax=1.2, reject = reject, baseline = (-0.1,0.))
    epochs.append(ep_cond)
    evoked.append(ep_cond.average())
    #evoked[cond].plot(picks=picks,titles=conds[cond])

conds.extend(['12AB','12BA', '8AB', '8BA'])
ev_combos=[[2,4],[3,5],[6,8],[7,9]]
#ev_combos = [[3,5],[4,6],[7,9],[8,10]]

for it, cnd in enumerate(range(10,14)):
    ep_cnd = mne.Epochs(raw,eves_AB,list(np.array(ev_combos[it])+1),tmin=-0.3,tmax=1.2, reject = reject, baseline = (-0.1,0.))
    epochs.append(ep_cnd)
    evoked.append(ep_cnd.average())
    #evoked[cnd].plot(picks=picks,titles=conds[cnd])

# Also get whole interval without baselining each interval
conds.extend(['12', '8'])
ep_cnd = mne.Epochs(raw,eves,1,tmin=-0.3,tmax=5.5, reject = reject, baseline = (-0.1,0.))
epochs.append(ep_cnd)
evoked.append(ep_cnd.average())

ep_cnd = mne.Epochs(raw,eves,2,tmin=-0.3,tmax=5.5, reject = reject, baseline = (-0.1,0.))
epochs.append(ep_cnd)
evoked.append(ep_cnd.average())

#%% Make Plots outside of MNE
combos_comp = [[1,0], [12,10], [13,11]] #Reversed coz of the way the stimulus is saved -- Here, event_ID 1 is for 12 tone complex, and 2 is for 8 tone complex
comp_labels = ['Onset', 'Incoherent to Coherent', 'Coherent to Incoherent']
t=epochs[0].times

fig, ax = plt.subplots(3,1,sharex=True)

for cnd in range(len(combos_comp)):
    # cz_ep_8 = epochs[combos_comp[cnd][0]].get_data()[:,8,:]
    # cz_mean_8 = cz_ep_8.mean(axis=0)
    # cz_sem_8 = cz_ep_8.std(axis=0) / np.sqrt(cz_ep_8.shape[0])

    cz_ep_12 = epochs[combos_comp[cnd][1]].get_data()[:,7,:]
    cz_mean_12 = cz_ep_12.mean(axis=0)
    cz_sem_12 = cz_ep_12.std(axis=0) / np.sqrt(cz_ep_12.shape[0])
    
    # ep_mastoid_8 = epochs[combos_comp[cnd][0]].get_data()[:,37,:]                     #Mastoid -EXG3
    # ep_vertex_8 = epochs[combos_comp[cnd][0]].get_data()[:,35,:]                      #Vertex -EXG4
    # # ep_ground_8 = epochs[combos_comp[cnd][0]].get_data()[:,37,:]                      #Ground - EXG
    # ep_subderm_8 = ep_vertex_8 - ep_mastoid_8 #Inverting mastoid and non-inverting vertex 
    # ep_mean_subderm_8 = ep_subderm_8.mean(axis=0)
    # ep_sem_subderm_8 = ep_subderm_8.std(axis=0) / np.sqrt(ep_subderm_8.shape[0])
    
    ep_mastoid_12 = epochs[combos_comp[cnd][1]].get_data()[:,34,:]                      #Mastoid -EXG3
    ep_vertex_12 = epochs[combos_comp[cnd][1]].get_data()[:,35,:]                       #Vertex -EXG4
    # ep_ground_12 = epochs[combos_comp[cnd][1]].get_data()[:,37,:]                       #Ground - EXG
    ep_subderm_12 = ep_vertex_12 - ep_mastoid_12 #Inverting mastoid and non-inverting vertex 
    ep_mean_subderm_12 = ep_subderm_12.mean(axis=0)
    ep_sem_subderm_12 = ep_subderm_12.std(axis=0) / np.sqrt(ep_subderm_12.shape[0])

    # ax[cnd].plot(t,cz_mean_8,label='8_EEG Cap')
    # ax[cnd].fill_between(t,cz_mean_8 - cz_sem_8, cz_mean_8 + cz_sem_8,alpha=0.5)

    ax[cnd].plot(t,cz_mean_12,label='12_EEG Cap')
    ax[cnd].fill_between(t,cz_mean_12 - cz_sem_12, cz_mean_12 + cz_sem_12,alpha=0.5)
    
    # ax[cnd].plot(t,ep_mean_subderm_8,label='8_Subdermal')
    # ax[cnd].fill_between(t,ep_mean_subderm_8- ep_sem_subderm_8, ep_mean_subderm_8 + ep_sem_subderm_8,alpha=0.5)

    ax[cnd].plot(t, ep_mean_subderm_12,label='12_Subdermal')
    ax[cnd].fill_between(t,ep_mean_subderm_12 -  ep_sem_subderm_12, ep_mean_subderm_12 +  ep_sem_subderm_12,alpha=0.5)

    ax[cnd].set_title(comp_labels[cnd])
    ax[cnd].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax[cnd].set_ylim([-4*1e-6, 5*1e-6])

# ax.set_ylim([-4*1e-6, 7*1e-6])
ax[0].legend(prop={'size': 6})
ax[2].set_xlabel('Time(sec)')
ax[1].set_ylabel('Amplitude (\u03bcV)')
plt.suptitle(subj + '_Binding - Awake')
plt.show()

#plt.savefig(os.path.join(save_loc,subj + 'Awake_SBvsCap_All3.png'),format='png', dpi=300)

#%% Save Epochs -- Trying to extend conditions to save subdermal electrodes as well, try mat file saving here as well! 
save_indexes = [0,10,11,14]
conds_save = []
#epochs_save = [] #Testing out with 2 channels right now
#epochs_save = np.zeros(2,8)
epochs_8_save = []
epochs_mastoid = []         #mastoid
epochs_vertex = []         #vertex
evoked_save = [] #save 32 channel evkd response
# all_channels = (np.arange(1,38))

t_full = epochs[-1].times    #The last value, which is for the full time (5.5 seconds)

for si in save_indexes:
      conds_save.append(conds[si])
      evoked_save.append(evoked[si])
      epochs_8_save.append(np.array(epochs[si].get_data()[:,7,:])) 
      epochs_mastoid.append(epochs[si].get_data()[:,34,:])
      epochs_vertex.append(epochs[si].get_data()[:,35,:])
      # epochs_ground.append(epochs[si].get_data()[:,37,:])      # Only save epochs for each one of these channels
      # epochs_save.append((np.array(epochs[si].get_data()[:,8,:])))
      # epochs_save.append((np.array(epochs[si].get_data()[:,24,:])))
     
# epochs_save = np.array(epochs_save)
      
pickle_loc='C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Chin_Analysis/Chin_Pickles/'
# with open(os.path.join(pickle_loc, subj + '_Binding_Awake_2channels_Trial.pickle'),'wb') as file:
#     pickle.dump([t, t_full, conds_save, epochs_save, evoked_save],file)
    
# del epochs, evoked, evoked_save,epochs_save
    
with open(os.path.join(pickle_loc, subj + '_Binding_Awake_A8_SB_12Only.pickle'),'wb') as file:
    pickle.dump([t, t_full, conds_save, epochs_8_save, epochs_mastoid, epochs_vertex,evoked_save],file)

del epochs, evoked, evoked_save,epochs_8_save, epochs_mastoid, epochs_vertex

# mat_ids = dict(combos_comp = combos_comp, conds_save=conds_save, epochs_save = epochs_save, evoked_save = evoked_save)
# savemat(save_loc_mat + subj + '_Binding(300-800ms).mat', mat_ids)