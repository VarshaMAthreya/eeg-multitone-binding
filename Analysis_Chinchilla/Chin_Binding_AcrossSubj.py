# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:32:44 2022

@author: vmysorea
"""

###This script gives anesthetized and awake compilation. Anesthetized data had
#both 8 and 12 coherence conditions, whereas awake had only 12 condition.
#Modifications are done here for the same

import sys
sys.path.append('C:/Users/vmysorea/Documents/mne-python/')
sys.path.append('C:/Users/vmysorea/Documents/ANLffr/')
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

#data_loc ='D:/PhD/Data/Binding_Pickles/'
pickle_loc = 'C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Chin_Analysis/Chin_Pickles/'

save_loc =  'C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Chin_Analysis/Figures/'

Anesthetized_Subjects = ['Q414', 'Q417']
Awake_Subjects = ['Q414', 'Q414_1', 'Q414_2', 'Q415', 'Q419', 'Q412']

Subjects = ['Q414', ]         #To check for subdermal data extraction 

Anesthetized_epochs = []
Anesthetized_evoked = []
Awake_epochs = []
Awake_evoked = []

A_epochs=[]
A_evoked=[]
all_channels = np.arange(1.,38.)
#%% Load data
#For all sunjects, once I analyze it right, and make it uniform
for subject in Subjects:
    print('Loading awake ' + subject)
    with open(os.path.join(pickle_loc,subject+'_Binding_Awake_Allchannels_Trial1.pickle'),"rb") as file:
        #[t, t_full, conds_save, epochs_save,evoked_save] = pickle.load(file)
        [t, t_full, conds_save, epochs_8_save, 
         epochs_35_save, epochs_36_save, epochs_37_save, evoked_save] = pickle.load(file)
    A_epochs.extend([epochs_8_save, epochs_35_save, epochs_36_save, epochs_37_save])
    #A_epochs.stack(epochs_save,0)
    # A_epochs.append(np.array(epochs_save))
    # print(np.array(epochs_save).shape)
    A_evoked.append(evoked_save)

A_epochs = np.array(A_epochs)


##Anesthetized - Coz 2 coherence conditions, so saved differently
for subject in Anesthetized_Subjects:
    print('Loading anesthetized ' + subject)
    with open(os.path.join(pickle_loc,subject+'_Binding_Anesthetized_A8.pickle'),"rb") as file:
        [t, t_full, conds_save, epochs_save,evoked_save] = pickle.load(file)
    Anesthetized_epochs.append(epochs_save)
    Anesthetized_evoked.append(evoked_save)

##Awake
for subject in Awake_Subjects:
    print('Loading awake ' + subject)
    with open(os.path.join(pickle_loc,subject+'_Binding_Awake_A8.pickle'),"rb") as file:
        [t_1, t_full_1, conds_save_1, epochs_save_1,evkd_save_1] = pickle.load(file)
    Awake_epochs.append(epochs_save_1)
    Awake_evoked.append(evkd_save_1)

#%% 32 Channel responses - Evoked response (So average of all channels)
#All channels 
evoked_12 = []
evoked_8 = []
for sub in range(len(Subjects)):
    evoked_12.append(A_evoked[sub][6].data)
    evoked_8.append(A_evoked[sub][7].data)


#Anesthetized
evoked_12 = []
evoked_20 = []
for sub in range(len(Anesthetized_Subjects)):
    evoked_12.append(Anesthetized_evoked[sub][6].data)
    evoked_20.append(Anesthetized_evoked[sub][7].data)

#Awake
evkd_12 = []
for sub in range(len(Awake_Subjects)):
    evkd_12.append(Awake_evoked[sub][3].data)

#%% Get evoked responses from epochs of channels saved 
#EEG cap - Mean of all channels
A_epochs=np.array(A_epochs)
A_8 = A_epochs[0:8]
channels =np.arange (1.,33.)
cz_ep = np.zeros((len(t),len(A_epochs),len(conds_save)-2))

for cond in range(len(conds_save)-2):        
    for channel in range(33):
        cz_ep[:,channel,cond] = A_epochs[channel][cond].mean(axis=0)  #Getting data from all scalp channels
        cz_ep_mean = cz_ep.mean(axis=1)
        cz_ep_sem = cz_ep.std(axis=1) / np.sqrt(cz_ep.shape[1])
        
for sub in range(len(Subjects)):
    cz_ep[:,sub,cond] = A_epochs[0][sub][cond].mean(axis=0)   

#%%EEG Cap - Getting A_8 data only when the pickle is saved with just A8 data 
cz_ep = np.zeros((len(t),len(Subjects),len(conds_save)-2))

for sub in range(len(Subjects)):
    for cond in range(len(conds_save)-2):        
        # for channel in range(len(A_epochs)):
            cz_ep[:,sub,cond] = A_epochs[0][sub][cond].mean(axis=0)          #Getting data from channel A8
            # cz_ep_mean = cz_ep.mean(axis=0)
            # cz_ep_sem = cz_ep.std(axis=0) / np.sqrt(cz_ep.shape[0])

#%%Getting data from subdermal electrodes - when saved separately in pickles 
mastoid_ep = np.zeros((len(t),len(Subjects),len(conds_save)-2))

for sub in range(len(Subjects)):
    for cond in range(len(conds_save)-2):        
        for channel in range(len(A_epochs)):
            mastoid_ep[:,sub,cond] = A_epochs[1][sub][cond].mean(axis=0)         
            # mastoid_ep_mean =mastoid_ep.mean(axis=1)
            #mastoid_ep_sem = mastoid_ep.std(axis=0) / np.sqrt(mastoid_ep.shape[0])
            
vertex_ep = np.zeros((len(t),len(Subjects),len(conds_save)-2))

for sub in range(len(Subjects)):
    for cond in range(len(conds_save)-2):        
        for channel in range(len(A_epochs)):
            vertex_ep[:,sub,cond] = A_epochs[2][sub][cond].mean(axis=0)          #Getting data from channel A8
            vertex_ep_mean =vertex_ep.mean(axis=0)
            vertex_ep_sem = vertex_ep.std(axis=0) / np.sqrt(vertex_ep.shape[0])
            
for sub in range(len(Subjects)):
        ep_subderm_12 = mastoid_ep - vertex_ep #Inverting mastoid and non-inverting vertex 
        #ep_mean_subderm_12 = ep_subderm_12.mean(axis=0)
        # ep_sem_subderm_12 = ep_subderm_12.std(axis=0) / np.sqrt(ep_subderm_12.shape[0])

#%%Anesthetized data 
anes_evoked_cz = np.zeros((len(t),len(Anesthetized_Subjects),len(conds_save)-2))

for sub in range(len(Anesthetized_Subjects)):
    for cond in range(len(conds_save)-2):
        anes_evoked_cz[:,sub,cond] = Anesthetized_epochs[sub][cond].mean(axis=0)

#%%Awake data
awake_evkd_cz = np.zeros((len(t_1),len(Awake_Subjects),len(conds_save_1)-1))

for sub in range(len(Awake_Subjects)):
    for cond in range(len(conds_save_1)-1):
        awake_evkd_cz[:,sub,cond] = Awake_epochs[sub][cond].mean(axis=0)

#%% Plot Average response across Subjects (With baseline)

#Conds:
#   0 = 12 Onset
#   1 = 8 Onset
#   3 = 12AB
#   4 = 12BA
#   5 = 8AB
#   6 = 8BA

cond_bind = ['12 Onset', '8 Onset', '12AB', '12BA', '8AB', '8BA', '12 all','8 all'] ##Reversed for chins
conds_comp = [[0,1], [2,4], [3,5]]
labels = ['Onset', 'Incoherent to Coherent', 'Coherent to Incoherent']

cond_bind_anes = ['8 Onset', '12 Onset', '8AB', '8BA', '12AB', '12BA', '8 all','12 all']
conds_comp_anes = [[1], [4], [5]]
labels = ['Onset', 'Incoherent to Coherent', 'Coherent to Incoherent']

cond_bind_awake = ['12 Onset', '12AB', '12BA']
conds_comp_awake = [[0], [5], [6]]
labels = ['Onset', 'Incoherent to Coherent', 'Coherent to Incoherent']
      
#%%Awake
fig,ax = plt.subplots(3,1,sharex=True)
for cnd in range(len(conds_comp_awake)):
    awake_onset12_mean = (awake_evkd_cz[:,:,cnd]*1e6).mean(axis=1)
    awake_onset12_sem = (awake_evkd_cz[:,:,cnd]*1e6).std(axis=1) / np.sqrt(awake_evkd_cz.shape[1])

    ax[cnd].plot(t,awake_onset12_mean,label='12_Awake')
    ax[cnd].fill_between(t_1,awake_onset12_mean - awake_onset12_sem,
                          awake_onset12_mean + awake_onset12_sem,alpha=0.5)

    ax[cnd].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax[cnd].tick_params(labelsize=12)

ax[0].legend(fontsize=12)
ax[0].set_title('Onset', loc='center', fontsize=10)
ax[1].set_title('Incoherent to Coherent', loc='center', fontsize=10)
ax[2].set_title('Coherent to Incoherent', loc='center', fontsize=10)
ax[2].set_xlabel('Time (s)',fontsize=14)
ax[1].set_ylabel('\u03bcV',fontsize=14)
ax[2].set_xlim([-0.050,1])
ax[2].set_xticks([0,0.2,0.4,0.6,0.8,1.0])
#ax[2].set_ylabel('$\mu$V')
fig.suptitle('Binding Across Chins')

plt.show()



#%%Anesthetized vs Awake
fig,ax = plt.subplots(3,1,sharex=True)
for cnd in range(len(conds_comp_anes)):
    anes_onset12_mean = (anes_evoked_cz[:,:,cnd]*1e6).mean(axis=1)
    anes_onset12_sem = (anes_evoked_cz[:,:,cnd]*1e6).std(axis=1) / np.sqrt(anes_evoked_cz.shape[1])

    ax[cnd].plot(t,anes_onset12_mean,label='12_Anesthetized')
    ax[cnd].fill_between(t,anes_onset12_mean - anes_onset12_sem,
                         anes_onset12_mean + anes_onset12_sem,alpha=0.5)
    
#Awake
# for cnd in range(len(conds_comp_awake)):
#     awake_onset12_mean = (awake_evkd_cz[:,:,cnd]*1e6).mean(axis=1)
#     awake_onset12_sem = (awake_evkd_cz[:,:,cnd]*1e6).std(axis=1) / np.sqrt(awake_evkd_cz.shape[1])

#     ax[cnd].plot(t,awake_onset12_mean,label='12_Awake')
#     ax[cnd].fill_between(t_1,awake_onset12_mean - awake_onset12_sem,
#                          awake_onset12_mean + awake_onset12_sem,alpha=0.5)

#     ax[cnd].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
#     ax[cnd].tick_params(labelsize=12)

ax[0].legend(fontsize=12)
ax[0].set_title('Onset', loc='center', fontsize=10)
ax[1].set_title('Incoherent to Coherent', loc='center', fontsize=10)
ax[2].set_title('Coherent to Incoherent', loc='center', fontsize=10)
ax[2].set_xlabel('Time (s)',fontsize=14)
ax[1].set_ylabel('\u03bcV',fontsize=14)
ax[2].set_xlim([-0.050,1])
ax[2].set_xticks([0,0.2,0.4,0.6,0.8,1.0])
#ax[2].set_ylabel('$\mu$V')
fig.suptitle('Binding Across Chins')

plt.show()

#%%Plot across anesthetized and awake conditions - Using this now!!!

anes_onset12_mean = (anes_evoked_cz[:,:,1]*1e6).mean(axis=1)
anes_onset12_sem = (anes_evoked_cz[:,:,1]*1e6).std(axis=1) / np.sqrt(anes_evoked_cz.shape[1])

anes_AB12_mean = (anes_evoked_cz[:,:,4]*1e6).mean(axis=1)
anes_AB12_sem = (anes_evoked_cz[:,:,4]*1e6).std(axis=1) / np.sqrt(anes_evoked_cz.shape[1])

anes_BA12_mean = (anes_evoked_cz[:,:,5]*1e6).mean(axis=1)
anes_BA12_sem = (anes_evoked_cz[:,:,5]*1e6).std(axis=1) / np.sqrt(anes_evoked_cz.shape[1])

awake_onset12_mean = (awake_evkd_cz[:,:,0]*1e6).mean(axis=1)
awake_onset12_sem = (awake_evkd_cz[:,:,0]*1e6).std(axis=1) / np.sqrt(awake_evkd_cz.shape[1])

awake_AB12_mean = (awake_evkd_cz[:,:,1]*1e6).mean(axis=1)
awake_AB12_sem = (awake_evkd_cz[:,:,1]*1e6).std(axis=1) / np.sqrt(awake_evkd_cz.shape[1])

awake_BA12_mean = (awake_evkd_cz[:,:,2]*1e6).mean(axis=1)
awake_BA12_sem = (awake_evkd_cz[:,:,2]*1e6).std(axis=1) / np.sqrt(awake_evkd_cz.shape[1])

fig, ax = plt.subplots(3, 1, sharex=True, constrained_layout=True)
ax[0].errorbar(t, anes_onset12_mean, yerr=anes_onset12_sem,
               color='purple', linewidth=2, ecolor='thistle',
               label='Anesthetized (N=2)')
ax[0].errorbar(t, awake_onset12_mean, yerr=awake_onset12_sem,
               color='green', linewidth=2, ecolor='palegreen',
               label='Awake (N=3)')
ax[1].errorbar(t, anes_AB12_mean, yerr=anes_AB12_sem,
               color='purple', linewidth=2, ecolor='thistle')
ax[1].errorbar(t, awake_AB12_mean, yerr=awake_AB12_sem,
               color='green', linewidth=2, ecolor='palegreen')
ax[2].errorbar(t, anes_BA12_mean, yerr=anes_BA12_sem,
               color='purple', linewidth=2, ecolor='thistle')
ax[2].errorbar(t, awake_BA12_mean, yerr=awake_BA12_sem,
               color='green', linewidth=2, ecolor='palegreen')
ax[0].set_title('Onset', loc='center', fontsize=10)
ax[1].set_title('Incoherent to Coherent', loc='center', fontsize=10)
ax[2].set_title('Coherent to Incoherent', loc='center', fontsize=10)
plt.xlim([-0.1, 0.6])
ax[0].legend(prop={'size': 6})
plt.xlabel('Time (in seconds)')
fig.text(0.01, 0.5, 'Amplitude(\u03bcV)', va='center', rotation='vertical')
plt.suptitle('Binding across Chinchillas')
plt.rcParams["figure.figsize"] = (5.5, 5)
plt.tight_layout()
plt.show()

plt.savefig(save_loc+'AllChins_AnesvsAwake_baselined_0.6s.png', dpi=300)

###Plotting just the onsets

plt.errorbar(t, anes_onset12_mean, yerr=anes_onset12_sem,
              color='purple', linewidth=2, ecolor='thistle',
              label='Anesthetized')
plt.errorbar(t, awake_onset12_mean, yerr=awake_onset12_sem,
               color='green', linewidth=2, ecolor='palegreen',
               label='Awake')
plt.legend()
plt.xlabel('Time (in seconds)')
plt.ylabel('Amplitude (\u03bcV)')
plt.title('Onset across chinchillas')
plt.rcParams["figure.figsize"] = (5.5, 5)
plt.xlim(-0.1,1.1)
#plt.ylim(-4,4)
plt.tight_layout()
plt.show()

plt.savefig(save_loc+'AllChins_AnesvsAwake_Onsets.png', dpi=300)

#%%Plotting A8 and subdermal 

cond_bind = ['12 Onset', '12AB', '12BA']
conds_comp = [[0], [2], [3]]
labels = ['Onset', 'Incoherent to Coherent', 'Coherent to Incoherent']

cz_onset12_mean = (cz_ep[:,:,0]*1e6).mean(axis=1)
cz_onset12_sem = (cz_ep[:,:,0]*1e6).std(axis=1) / np.sqrt(cz_ep.shape[1])

cz_AB12_mean = (cz_ep[:,:,2]*1e6).mean(axis=1)
cz_AB12_sem = (cz_ep[:,:,2]*1e6).std(axis=1) / np.sqrt(cz_ep.shape[1])

cz_BA12_mean = (cz_ep[:,:,3]*1e6).mean(axis=1)
cz_BA12_sem = (cz_ep[:,:,3]*1e6).std(axis=1) / np.sqrt(cz_ep.shape[1])

subderm_onset12_mean = (ep_subderm_12[:,:,0]*1e6).mean(axis=1)
subderm_onset12_sem = (ep_subderm_12[:,:,0]*1e6).std(axis=1) / np.sqrt(ep_subderm_12.shape[1])

subderm_AB12_mean = (ep_subderm_12[:,:,2]*1e6).mean(axis=1)
subderm_AB12_sem = (ep_subderm_12[:,:,2]*1e6).std(axis=1) / np.sqrt(ep_subderm_12.shape[1])

subderm_BA12_mean = (ep_subderm_12[:,:,3]*1e6).mean(axis=1)
subderm_BA12_sem = (ep_subderm_12[:,:,3]*1e6).std(axis=1) / np.sqrt(ep_subderm_12.shape[1])

fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, constrained_layout=True)
ax[0].errorbar(t, cz_onset12_mean, yerr=cz_onset12_sem,
               color='purple', linewidth=2, ecolor='thistle',
               label='A8 EEG Cap')
ax[0].errorbar(t, subderm_onset12_mean, yerr=subderm_onset12_sem,
               color='green', linewidth=2, ecolor='palegreen',
               label='Subdermal')
ax[1].errorbar(t, cz_AB12_mean, yerr=cz_AB12_sem,
               color='purple', linewidth=2, ecolor='thistle')
ax[1].errorbar(t, subderm_AB12_mean, yerr=subderm_AB12_sem,
               color='green', linewidth=2, ecolor='palegreen')
ax[2].errorbar(t, cz_BA12_mean, yerr=cz_BA12_sem,
               color='purple', linewidth=2, ecolor='thistle')
ax[2].errorbar(t, subderm_BA12_mean, yerr=subderm_BA12_sem,
               color='green', linewidth=2, ecolor='palegreen')
ax[0].set_title('Onset', loc='center', fontsize=10)
ax[1].set_title('Incoherent to Coherent', loc='center', fontsize=10)
ax[2].set_title('Coherent to Incoherent', loc='center', fontsize=10)
plt.xlim([-0.1, 0.6])
ax[0].legend(prop={'size': 6})
plt.xlabel('Time (in seconds)')
fig.text(0.01, 0.5, 'Amplitude(\u03bcV)', va='center', rotation='vertical')
plt.suptitle('Binding across Awake Chinchillas')
plt.rcParams["figure.figsize"] = (5.5, 5)
plt.tight_layout()
plt.show()

#%% Plotting across 5 seconds

#t_f = np.arange(-0.3,5.5+1/4096,1/4096)
t_f = t_full

A_evkd_f = np.zeros((len(t_f),len(Subjects),2))

for sub in range(len(Subjects)):
    for cond in range(2):
        A_evkd_f[:,sub,cond] = A_epochs[sub][6+cond].mean(axis=0)


sub_mean = A_evkd_f[:,:,0].mean(axis=1) * 1e6
sub_sem = (A_evkd_f[:,:,0]*1e6).std(axis=1) / np.sqrt(len(Subjects))

plt.figure()
plt.plot(t_f,sub_mean,label='12')
plt.fill_between(t_f, sub_mean-sub_sem,sub_mean+sub_sem,alpha=0.5)

sub_mean = A_evkd_f[:,:,1].mean(axis=1) * 1e6
sub_sem = (A_evkd_f[:,:,1]*1e6).std(axis=1) / np.sqrt(len(Subjects))

plt.plot(t_f,sub_mean, label='20')
plt.fill_between(t_f, sub_mean-sub_sem,sub_mean+sub_sem,alpha=0.5)

plt.xlabel('Time (s)',fontsize=14)
plt.ylabel('\u03bcV',fontsize=14)
plt.tick_params(labelsize=12)
plt.legend(fontsize=12)

plt.savefig(save_loc+'All_12vs20_noBaseline.png', dpi=300)