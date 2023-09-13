# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 22:54:14 2023

@author: vmysorea
"""
#Binding across chins - Only one condition (12 tone - Better one)

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

Subjects = ['Q414', 'Q416', 'Q418']

A_epochs = []
A_evkd = []

#%% Load data

for subject in Subjects:
    print('Loading ' + subject)
    with open(os.path.join(pickle_loc,subject+'_Binding_Awake_A8.pickle'),"rb") as file:
        [t, t_full, conds_save, epochs_save,evkd_save] = pickle.load(file)
    A_epochs.append(epochs_save)
    A_evkd.append(evkd_save)

#%% 32 Channel responses

evkd_12 = []
for sub in range(len(Subjects)):
    evkd_12.append(A_evkd[sub][0].data)

#%% Get evoked responses

A_evkd_cz = np.zeros((len(t),len(Subjects),len(conds_save)))

for sub in range(len(Subjects)):
    for cond in range(len(conds_save)):
        A_evkd_cz[:,sub,cond] = A_epochs[sub][cond].mean(axis=0)

#%% Plot Average response across chins (With baseline)

cond_bind = ['12 Onset', '12AB', '12BA']

conds_comp = [[0], [5], [6]]
labels = ['Onset', 'Incoherent to Coherent', 'Coherent to Incoherent']

fig,ax = plt.subplots(3,1,sharex=True)
for cnd in range(len(conds_comp)):
    onset12_mean = (A_evkd_cz[:,:,cnd]*1e6).mean(axis=1)
    onset12_sem = (A_evkd_cz[:,:,cnd]*1e6).std(axis=1) / np.sqrt(A_evkd_cz.shape[1])

    ax[cnd].plot(t,onset12_mean,label='12')
    ax[cnd].fill_between(t,onset12_mean - onset12_sem, onset12_mean + onset12_sem,alpha=0.5)

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
fig.suptitle('Binding Across Chins - Awake (N=3)')

plt.show()

#%%New plot across conditions
onset12_mean = (A_evkd_cz[:,:,0]*1e6).mean(axis=1)
onset12_sem = (A_evkd_cz[:,:,0]*1e6).std(axis=1) / np.sqrt(A_evkd_cz.shape[1])

AB12_mean = (A_evkd_cz[:,:,1]*1e6).mean(axis=1)
AB12_sem = (A_evkd_cz[:,:,1]*1e6).std(axis=1) / np.sqrt(A_evkd_cz.shape[1])

BA12_mean = (A_evkd_cz[:,:,2]*1e6).mean(axis=1)
BA12_sem = (A_evkd_cz[:,:,2]*1e6).std(axis=1) / np.sqrt(A_evkd_cz.shape[1])

fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, constrained_layout=True)
ax[0].errorbar(t, onset12_mean, yerr=onset12_sem,
               color='darkblue', linewidth=2, ecolor='lightsteelblue')
ax[1].errorbar(t, AB12_mean, yerr=AB12_sem,
               color='purple', linewidth=2, ecolor='thistle')
ax[0].set_title('Onset', loc='center', fontsize=10)
ax[2].errorbar(t, BA12_mean, yerr=BA12_sem,
               color='green', linewidth=2, ecolor='palegreen')
ax[1].set_title('Incoherent to Coherent', loc='center', fontsize=10)
ax[2].set_title('Coherent to Incoherent', loc='center', fontsize=10)
plt.xlim([-0.1, 1.0])
ax[0].legend(prop={'size': 6})
plt.xlabel('Time (in seconds)')
fig.text(0.0001, 0.5, 'Amplitude', va='center', rotation='vertical')
plt.suptitle('Binding in Awake Chinchillas (N=3)')
plt.rcParams["figure.figsize"] = (5.5, 5)
plt.tight_layout()
plt.show()


plt.savefig(save_loc + 'AwakeBinding_All3Cond_N3.png', dpi=300)

#%% Plot only the onset

onset12_mean = (A_evkd_cz[:,:,1]*1e6).mean(axis=1)
onset12_sem = (A_evkd_cz[:,:,1]*1e6).std(axis=1) / np.sqrt(A_evkd_cz.shape[1])

#plt.plot(t, onset12_mean, label='12 Condition, N=982 trials')
plt.errorbar(t, onset12_mean, yerr=onset12_sem,
              color='purple', linewidth=2, ecolor='thistle')
plt.legend('N=3')
plt.xlabel('Time (in seconds)')
plt.ylabel('Amplitude (\u03bcV)')
plt.title('Onset across Awake Chins(982 trials)')
plt.rcParams["figure.figsize"] = (5.5, 5)
plt.xlim(-0.1,1.1)
plt.tight_layout()
plt.show()

plt.savefig(save_loc + 'Awake_Onset_A8_12Only (N=3).png', dpi=300)

#%% Plotting 300-800 ms
t1 = t>=0.3
t2 = t<=0.8
t3 = np.array([t2[i] and t1[i] for i in range(len(t1))])
#t3= np.logical_and(t1,t2) #Subtracting t2-t1

for jj in range(3):
     cnd1 = conds_comp[jj][0]
     cnd2 = conds_comp[jj][1]

     for subject in range(len(Subjects)):

        cz_12_mean = (A_evkd_cz[:,:,cnd1]*1e6)
        cz_12_t3 = cz_12_mean[:,t3]
        cz_12_t3_avg = abs(cz_12_t3).mean()

        cz_20_mean = (A_evkd_cz[:,:,cnd2]*1e6).mean(axis=1)
        cz_20_t3 = cz_20_mean[:,t3]
        cz_20_t3_avg = cz_20_t3.mean()
