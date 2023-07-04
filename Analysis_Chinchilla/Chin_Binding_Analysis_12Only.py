# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 17:51:19 2023

@author: vmysorea
"""
#%%Initializing...
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

#raw.crop(tmin=100,tmax=175,include_tmax=True)
#raw.plot()
#%% Heart beat artifact rej
# heartbeat = find_ecg_events(raw, ch_name='EXG5')
# raw.plot(events=heartbeat)

# epochs_blinks = mne.Epochs(raw,heartbeat, event_id=999, baseline=(-0.25, 0.25),
#                         reject=dict(eeg=500e-6), tmin=-0.25, tmax=0.25)

# blink_proj = compute_proj_epochs(epochs_blinks, n_eeg=1)

# raw.add_proj(blink_proj)  # Adding the n=blink proj to the data --
# # removal of blinks

# raw.plot_projs_topomap()

#%% Trying out different artifact rejection methods here
#raw.plot(remove_dc=True)

# %% Filtering
for subj in subjlist:
    raw.filter(1., 20.)
    raw.info
    
####To see for alpha -ARO 
# chans = ['A1','A2','A23','A8','A30','A15','A22','A7','A18','A3']
# fs=raw.info['sfreq']
# dur_sec = np.array([790,800])
# start,stop=(dur_sec*fs).astype(int)
# crop1=raw[chans,start:stop]
# y_off=np.linspace(0.0016,0,10)
# y=crop1[0].T+y_off
# plt.plot(crop1[1],(crop1[0].T+y_off), color='black')
# # plt.yticks(ticks=np.arange(len(y)),labels=chans)
# plt.show()

# mat_ids=dict(chans=chans,fs=fs,start=start,stop=stop,raw=crop1)
# savemat(save_loc + 'alpha_raw.mat', mat_ids)

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

#%%Topo layout

# box = (0,1,0,1);

# sx =.1; 
# sy =.1;

# row_1 = [.5,1,sx,sy];

# row_2 = np.arange(1/6,5/6,1/6);
# row_2 = np.vstack(row_2);
# row_2 = np.hstack([row_2,5/6*np.vstack(np.ones(len(row_2))),np.vstack(np.ones(len(row_2)))*sx,np.vstack(np.ones(len(row_2)))*sy]);

# row_35_x = np.arange(1/8,1,1/8);
# row_35_x = np.vstack(row_35_x);
# row_3 = np.hstack([row_35_x,4/6*np.vstack(np.ones(len(row_35_x))),np.vstack(np.ones(len(row_35_x)))*sx,np.vstack(np.ones(len(row_35_x)))*sy])
# row_4 = np.hstack([row_35_x,3/6*np.vstack(np.ones(len(row_35_x))),np.vstack(np.ones(len(row_35_x)))*sx,np.vstack(np.ones(len(row_35_x)))*sy])
# row_5 = np.hstack([row_35_x,2/6*np.vstack(np.ones(len(row_35_x))),np.vstack(np.ones(len(row_35_x)))*sx,np.vstack(np.ones(len(row_35_x)))*sy])

# row_6 = np.arange(1/6,5/6,1/6);
# row_6 = np.vstack(row_6);
# row_6 = np.hstack([row_6,1/6*np.vstack(np.ones(len(row_6))),np.vstack(np.ones(len(row_6)))*sx,np.vstack(np.ones(len(row_6)))*sy]);

# pos = np.vstack([row_1,row_2,row_3,row_4,row_5,row_6]);

# #vertical and horizontal manual tweak
# pos[:,1] -= .1;
# pos[:,0] -= .05;


# names = ["A1","A4","A3","A2","A18","A19","A5","A6","A7","A23","A22","A21","A20","A11","A10","A9","A8","A24","A25","A26","A12","A13","A14","A30","A29","A28","A27","A17","A16","A15","A31","A32"];
# ids = [1,4,3,2,18,19,5,6,7,23,22,21,20,11,10,9,8,24,25,26,12,13,14,30,29,28,27,17,16,15,31,32];
# #print(pos)


# kind = 'biosemi';

# plt.rcParams.update({'lines.linewidth': 1})
# plt.rcParams.update({'figure.figsize': (6,6)})
# chin_cap_layout = mne.channels.Layout.__new__(mne.channels.Layout)
# chin_cap_layout.__init__(box,pos,names,ids,kind);
# chin_cap_layout.plot(show_axes = True);

# topo_fig = plt.figure();
# topo_fig.clear();
# plt.rcParams.update({'figure.figsize': (4,4)})
# plt.rcParams.update({'lines.linewidth': 3})
# topo_fig = plt.figure(dpi = 300)
# ax = plt.gca();
# topo_fig = evoked.plot_topo(layout = chin_cap_layout,ylim = dict(eeg=[-1,1.]),legend=True, axes = ax);
# topo_fig.show()
# plt.show()

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
#%% Saving epochs - Pickles

save_indexes = [0,5,6,7]
conds_save = []
# epochs_save = []
epochs_8_save = []
epochs_mastoid = []         #mastoid
epochs_vertex = []         #vertex       
evoked_save = [] #save 32 channel evkd response

t_full = epochs[-1].times

for si in save_indexes:
      conds_save.append(conds[si])
      evoked_save.append(evkd[si])
      epochs_8_save.append(np.array(epochs[si].get_data()[:,7,:])) 
      epochs_mastoid.append(epochs[si].get_data()[:,34,:])
      epochs_vertex.append(epochs[si].get_data()[:,35,:])    # Only save epochs for each one of these channels
      
with open(os.path.join(pickle_loc, subj + '_Binding_Awake_A8_SB.pickle'),'wb') as file:
    pickle.dump([t, t_full, conds_save, epochs_8_save, epochs_mastoid, epochs_vertex, evoked_save],file)
    
del epochs, evoked, evoked_save,epochs_8_save, epochs_mastoid, epochs_vertex

# pickle_loc='C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Chin_Analysis/Chin_Pickles/'
# with open(os.path.join(pickle_loc,subj+'_Binding_Awake_A8.pickle'),'wb') as file:
#     pickle.dump([t, t_full, conds_save, epochs_save,evkd_save],file)

# del epochs, evkd, evkd_save,epochs_save