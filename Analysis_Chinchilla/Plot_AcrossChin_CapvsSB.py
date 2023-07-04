# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 18:28:41 2023

@author: vmysorea
"""
#%%Initializing..
import sys
sys.path.append('C:/Users/vmysorea/Documents/mne-python/')
sys.path.append('C:/Users/vmysorea/Documents/ANLffr/')
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy.io import savemat

plt.switch_backend('QT5Agg')  # Making the plots interactive (Scrollable)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# Defining the dimensions and quality of figures
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 120

pickle_loc = 'C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Chin_Analysis/Chin_Pickles/Awake_SB_Cap_12Only/1-20Hz/'
save_loc =  'C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Chin_Analysis/Figures/'

Subjects = ['Q419', 'Q412','Q414_1', 'Q414_2', 'Q415', 'Q412']

#Not so great -- 'Q414'
A_epochs=[]
A_evoked=[]
epochs_8=[]
epochs_mastoid=[]
epochs_vertex=[]

#%% Load data
#For all subjects, once I analyze it right, and make it uniform
for subject in Subjects:
    print('Loading awake ' + subject)
    with open(os.path.join(pickle_loc,subject+'_Binding_Awake_A8_SB_12Only.pickle'),"rb") as file:
        [t, t_full, conds_save, epochs_8_save, 
         epochs_mastoid_save, epochs_vertex_save, evoked_save] = pickle.load(file)
    # A_epochs.extend([epochs_8_save, epochs_35_save, epochs_36_save, epochs_37_save])
    A_evoked.append(evoked_save)
    epochs_8.append(epochs_8_save)
    epochs_mastoid.append(epochs_mastoid_save)
    epochs_vertex.append(epochs_vertex_save)

#%% 32 Channel Evoked responses (So average of all channels)
#All channels 
# evoked_12 = []
# evoked_8 = []
# for sub in range(len(Subjects)):
#     evoked_12.append(A_evoked[sub][6].data)
#     evoked_8.append(A_evoked[sub][7].data)
    
#All channels in 12 only condition
evoked_12 = []
for sub in range(len(Subjects)):
    evoked_12.append(A_evoked[sub][3].data)
   
    
#%%EEG Cap - Getting A_8 data only when the pickle is saved with just A8 data 
cz_ep = np.zeros((len(t),len(Subjects),len(conds_save)-1))

for sub in range(len(Subjects)):
    for cond in range(len(conds_save)-1):        
        cz_ep[:,sub,cond] = epochs_8[sub][cond].mean(axis=0)          #Getting data from channel A8

#%%Getting data from subdermal electrodes - when saved separately in pickles 
mastoid_ep = np.zeros((len(t),len(Subjects),len(conds_save)-1))

for sub in range(len(Subjects)):
    for cond in range(len(conds_save)-1):        
        mastoid_ep[:,sub,cond] = epochs_mastoid[sub][cond].mean(axis=0)         
            
vertex_ep = np.zeros((len(t),len(Subjects),len(conds_save)-1))

for sub in range(len(Subjects)):
    for cond in range(len(conds_save)-1):        
        vertex_ep[:,sub,cond] = epochs_vertex[sub][cond].mean(axis=0)          #Getting data from channel A8
            
for sub in range(len(Subjects)):
        ep_subderm = vertex_ep - mastoid_ep   #Inverting mastoid and non-inverting vertex 

#%% Plot Average response across Subjects (With baseline)

#Conds:
#   0 = 12 Onset
#   1 = 8 Onset
#   2 = 12AB
#   3 = 12BA
#   4 = 8AB
#   5 = 8BA

cond_bind = ['12 Onset','12AB', '12BA'] ##Reversed for chins
conds_comp = [0,2,3]
labels = ['Onset', 'Incoherent to Coherent', 'Coherent to Incoherent']

cz_ep_mean = (cz_ep[:,:,0]*1e6).mean(axis=1)
cz_ep_sem = (cz_ep[:,:,0]*1e6).std(axis=1) / np.sqrt(cz_ep.shape[1])

cz_ep_AB12_mean = (cz_ep[:,:,1]*1e6).mean(axis=1)
cz_ep_AB12_sem = (cz_ep[:,:,1]*1e6).std(axis=1) / np.sqrt(cz_ep.shape[1])

cz_ep_BA12_mean = (cz_ep[:,:,2]*1e6).mean(axis=1)
cz_ep_BA12_sem = (cz_ep[:,:,2]*1e6).std(axis=1) / np.sqrt(cz_ep.shape[1])

ep_subderm_12_mean = (ep_subderm[:,:,0]*1e6).mean(axis=1)
ep_subderm_12_sem = (ep_subderm[:,:,0]*1e6).std(axis=1) / np.sqrt(ep_subderm.shape[1])

ep_subderm_AB12_mean = (ep_subderm[:,:,1]*1e6).mean(axis=1)
ep_subderm_AB12_sem = (ep_subderm[:,:,1]*1e6).std(axis=1) / np.sqrt(ep_subderm.shape[1])

ep_subderm_BA12_mean = (ep_subderm[:,:,2]*1e6).mean(axis=1)
ep_subderm_BA12_sem = (ep_subderm[:,:,2]*1e6).std(axis=1) / np.sqrt(ep_subderm.shape[1])

###Onset Only
fig, ax = plt.subplots(constrained_layout=True)
plt.plot(t, cz_ep_mean,label='A8_Cap', color='green')
plt.fill_between(t, cz_ep_mean - cz_ep_sem,
                      cz_ep_mean + cz_ep_sem,alpha=0.5, color='palegreen')
plt.plot(t, ep_subderm_12_mean,label='Subdermal', color='purple')
plt.fill_between(t, ep_subderm_12_mean - ep_subderm_12_sem,
                     ep_subderm_12_mean + ep_subderm_12_sem,alpha=0.5, color='thistle')
plt.xlabel('Time (in seconds)')
ax.set_ylabel('Amplitude(\u03bcV)')
# fig.text(0.0001, 0.5, 'Amplitude(\u03bcV)', va='center', rotation='vertical')
plt.suptitle('Binding - Onset Across Awake Chinchillas')
plt.rcParams["figure.figsize"] = (5.5, 5)
# plt.tight_layout()
plt.show()

plt.savefig(save_loc+'AllChins_SBvsA8_Onset.png', dpi=300)

#Saving to mat file
mat_ids_onset = dict(A8_mean=cz_ep_mean,A8_sem=cz_ep_sem,subderm_mean=ep_subderm_12_mean, subderm_sem = ep_subderm_12_sem, t=t)
save_loc = ('C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Chin_Analysis/')
savemat(save_loc + 'Onsets_SBvsA8.mat', mat_ids_onset)

##All 3 conditions
fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, constrained_layout=True)

ax[0].plot(t, cz_ep_mean,label='A8_Cap', color='green')
ax[0].fill_between(t, cz_ep_mean - cz_ep_sem,
                      cz_ep_mean + cz_ep_sem,alpha=0.5, color='palegreen')
ax[0].plot(t, ep_subderm_12_mean,label='Subdermal', color='purple')
ax[0].fill_between(t, ep_subderm_12_mean - ep_subderm_12_sem,
                     ep_subderm_12_mean + ep_subderm_12_sem,alpha=0.5, color='thistle')
ax[1].plot(t, cz_ep_AB12_mean, color='green')
ax[1].fill_between(t, cz_ep_AB12_mean - cz_ep_AB12_sem,
                     cz_ep_AB12_mean + cz_ep_AB12_sem,alpha=0.5, color='palegreen')
ax[1].plot(t, ep_subderm_AB12_mean, color='purple')
ax[1].fill_between(t, ep_subderm_AB12_mean - ep_subderm_AB12_sem,
                     ep_subderm_AB12_mean + ep_subderm_AB12_sem,alpha=0.5, color='thistle')
ax[2].plot(t, cz_ep_BA12_mean, color='green')
ax[2].fill_between(t, cz_ep_BA12_mean - cz_ep_BA12_sem,
                     cz_ep_BA12_mean + cz_ep_BA12_sem,alpha=0.5, color='palegreen')
ax[2].plot(t, ep_subderm_BA12_mean, color ='purple')
ax[2].fill_between(t, ep_subderm_BA12_mean - ep_subderm_BA12_sem,
                     ep_subderm_BA12_mean + ep_subderm_BA12_sem,alpha=0.5, color='thistle')

ax[0].set_title('Onset', loc='center', fontsize=10)
ax[1].set_title('Incoherent to Coherent', loc='center', fontsize=10)
ax[2].set_title('Coherent to Incoherent', loc='center', fontsize=10)
plt.xlim([-0.2, 1.1])
ax[0].legend(prop={'size': 6})
plt.xlabel('Time (in seconds)')
ax[1].set_ylabel('Amplitude(\u03bcV)')
# fig.text(0.00000001, 0.5, 'Amplitude(\u03bcV)', va='center', rotation='vertical')
plt.suptitle('Binding Across Awake Chinchillas')
plt.rcParams["figure.figsize"] = (5.5, 5)
#plt.tight_layout()
plt.show()

plt.savefig(save_loc+'AllChins_SBvsCap_baselined_1.png', dpi=300)

#%% Plotting across 5 seconds

t_f = t_full

#Getting responses A8 and subdermal electrodes for plotting full time (last condition=12)
cz_ep_full = np.zeros((len(t_f),len(Subjects),1))

for sub in range(len(Subjects)):
    for cond in range(1):        
        cz_ep_full[:,sub,cond] = epochs_8[sub][3+cond].mean(axis=0) 

mastoid_ep_full = np.zeros((len(t_f),len(Subjects),1))

for sub in range(len(Subjects)):
    for cond in range(1):        
        mastoid_ep_full[:,sub,cond] = epochs_mastoid[sub][3+cond].mean(axis=0)         
            
vertex_ep_full = np.zeros((len(t_f),len(Subjects),1))

for sub in range(len(Subjects)):
    for cond in range(1):        
        vertex_ep_full[:,sub,cond] = epochs_vertex[sub][3+cond].mean(axis=0)          #Getting data from channel A8
            
for sub in range(len(Subjects)):
        ep_subderm_full = vertex_ep_full - mastoid_ep_full   #Inverting mastoid and non-inverting vertex 

cz_mean = (cz_ep_full[:,:,0]).mean(axis=1)
cz_sem = (cz_ep_full[:,:,0]).std(axis=1) / np.sqrt(len(Subjects))

sb_mean =  (ep_subderm_full[:,:,0]).mean(axis=1)
sb_sem = (ep_subderm_full[:,:,0]).std(axis=1) / np.sqrt(len(Subjects))

plt.figure()
plt.plot(t_f,cz_mean,label='A8 EEG Cap', color='green')
plt.fill_between(t_f, cz_mean-cz_sem,cz_mean+cz_sem,alpha=0.5, color='palegreen')

plt.plot(t_f,sb_mean,label='Subdermal', color='purple')
plt.fill_between(t_f, sb_mean-sb_sem,sb_mean+sb_sem,alpha=0.5, color='thistle')

plt.xlabel('Time (s)',fontsize=14)
# plt.ylabel('Amplitude(\u03bcV)',fontsize=14)
plt.tick_params(labelsize=12)
plt.legend(fontsize=12)
plt.title('Binding across Chins (N=3) +2 sess of Q414')
plt.legend()
plt.show()

plt.savefig(save_loc+'AllChins_SBvsA8_noBaseline.png', dpi=300)