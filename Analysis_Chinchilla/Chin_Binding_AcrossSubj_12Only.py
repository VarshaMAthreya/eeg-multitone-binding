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


#%% Young Vs Old
#Plot to show only the incoherent to coherent -- Hari talk

conds_comp = [(2,2)]
labels = ['Incoherent to Coherent 20']
fig1,ax = plt.subplots(squeeze=False)

for jj in range(1):
    cnd1 = conds_comp[jj][0]
    cnd2 = conds_comp[jj][1]

    onset12_mean = A_evkd_cz[:,young,cnd1].mean(axis=1)
    onset12_sem = A_evkd_cz[:,young,cnd1].std(axis=1) / np.sqrt(A_evkd_cz[:,young,cnd1].shape[1])

    ax = ax.flatten()
    ax[jj].plot(t,onset12_mean,label='Young',color='tab:green')
    ax[jj].plot(t,onset12_sem, alpha=0.5, color='tab:green')
    #ax[jj].fill_between(t,onset12_mean - onset12_sem, onset12_mean + onset12_sem,alpha=0.5, color='tab:green')

    onset20_mean = A_evkd_cz[:,old,cnd2].mean(axis=1)
    onset20_sem = A_evkd_cz[:,old,cnd2].std(axis=1) / np.sqrt(A_evkd_cz[:,old,cnd2].shape[1])

    ax[jj].plot(t,onset20_mean,label='Old', color='tab:purple')
    ax[jj].fill_between(t,onset20_mean - onset20_sem, onset20_mean + onset20_sem,alpha=0.5,color='tab:purple')

    ax[jj].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax[jj].set_title(labels[jj])

ax[0].legend()
ax[0].set_xlabel('Time')

plt.show()


#%% Normalize Data to max of of abs of onset response

# for sub in range(len(Subjects)):
#      sub_norm = np.max(np.abs((A_epochs[sub][0].mean(axis=0) + A_epochs[sub][1].mean(axis=0)) / 2))
#      for cnd in range(len(cond_bind)):
#         A_epochs[sub][cnd] /= sub_norm

#%% 5 second durations

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


#%% Plot Full time for individuals

fig,ax = plt.subplots(int(np.ceil(len(Subjects)/2)),2,sharex=True, figsize=(14,12))
ax = np.reshape(ax,[int(len(Subjects))])

for sub in range(len(Subjects)):
    cnd1 = 6
    cnd2 = 7

    mean_12 = A_epochs[sub][cnd1].mean(axis=0)
    sem_12 = A_epochs[sub][cnd1].std(axis=0) / np.sqrt(A_epochs[sub][cnd1].shape[0])

    ax[sub].plot(t_f,mean_12)
    ax[sub].fill_between(t_f,mean_12 - sem_12, mean_12+sem_12,alpha=0.5)

    mean_20= A_epochs[sub][cnd2].mean(axis=0)
    sem_20 = A_epochs[sub][cnd2].std(axis=0) / np.sqrt(A_epochs[sub][cnd2].shape[0])

    ax[sub].plot(t_f,mean_20)
    ax[sub].fill_between(t_f,mean_20-sem_20,mean_20+sem_20,alpha=0.5)

#%% Plot all Onsets, AB, or BA

fig,ax = plt.subplots(int(np.ceil(len(Subjects)/2)),2,sharex=True,sharey=True,figsize=(14,12))
ax = np.reshape(ax,[int(len(Subjects))])

cnd_plot = 1
labels = ['Onset', 'AtoB', 'BtoA']
conds_comp = [[0,1], [2,4], [3,5]]

fig.suptitle(labels[cnd_plot])

for sub in range(len(Subjects)):
    cnd1 = conds_comp[cnd_plot][0]
    cnd2 = conds_comp[cnd_plot][1]

    ax[sub].set_title(Subjects[sub])

    sub_norm = np.max(np.abs((A_epochs[sub][0].mean(axis=0) + A_epochs[sub][1].mean(axis=0)) / 2))
    sub_norm = 1 # get rid of normalization in plot for now

    onset12_mean = (A_epochs[sub][cnd1]).mean(axis=0) / sub_norm
    onset12_sem = (A_epochs[sub][cnd1]).std(axis=0) / np.sqrt(A_epochs[sub][cnd1].shape[0])

    ax[sub].plot(t,onset12_mean,label='12')
    ax[sub].fill_between(t,onset12_mean - onset12_sem, onset12_mean + onset12_sem,alpha=0.5)

    onset20_mean = (A_epochs[sub][cnd2]).mean(axis=0) / sub_norm
    onset20_sem = (A_epochs[sub][cnd2]).std(axis=0) / np.sqrt(A_epochs[sub][cnd2].shape[0])

    ax[sub].plot(t,onset20_mean, label='20')
    ax[sub].fill_between(t,onset20_mean - onset20_sem, onset20_mean + onset20_sem,alpha=0.5)

    #ax[sub].plot(t,onset20_mean-onset12_mean,label='diff',color='k')

    #ax[sub].ticklabel_format(axis='y',style='sci',scilimits=(0,0))


ax[2].legend()
ax[len(Subjects)-1].set_xlabel('Time (sec)')
#ax[len(Subjects)-1].set_ylabel('Norm Amp')

plt.savefig(save_loc+ 'Asubj_' + labels[cnd_plot] + '.png', dpi=300)