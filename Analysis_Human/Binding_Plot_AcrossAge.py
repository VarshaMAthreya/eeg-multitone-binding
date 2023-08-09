# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 16:55:50 2022

@author: vmysorea
"""
import sys
sys.path.append('C:/Users/vmysorea/Documents/mne-python/')
sys.path.append('C:/Users/vmysorea/Documents/ANLffr/')
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import pickle
import os
#import scipy.io as sio
# from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler
#from scipy import signal
# import mne
#from mne import read_events, read_evokeds, Epochs, read_epochs
#from mne.io._digitization import _get_fid_coords
#from mne import io.digitization

data_loc ='D:/PhD/Data/Binding_Pickles/'
pickle_loc = 'C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Human_Analysis/Pickles/'

save_loc =  'C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Human_Analysis/Figures/'

### Haven't considered S105

# Subjects = ['S069','S072','S078','S088','S104','S105','S259','S260','S268', 'S269',
#           'S270', 'S271', 'S273', 'S274','S277','S279','S280','S281','S282','S284',
#             'S285', 'S290' ,'S291','S303','S305',
#            'S308','S310','S312']

Subjects = ['S273', 'S268', 'S269', 'S274', 'S282', 'S285',
           'S277', 'S279', 'S280','S259','S270', 'S271', 
           'S281','S290', 'S284', 'S305','S303','S288','S260',
           'S352', 'S341', 'S312', 'S347', 'S340','S078','S069',
           'S088','S342','S072','S308','S344',
           'S105','S345','S291','S310','S339']

#Subjects = ['S273', 'S268', 'S269', 'S274', 'S282', 'S285',
 #            'S259','S277', 'S279', 'S280','S270', 'S271', 'S281', 'S290', 'S308', 'S105', 'S291', 'S310' ]
# #YNH
#Subjects = ['S273', 'S268', 'S269', 'S274', 'S282', 'S285',
             #'S259','S277', 'S279', 'S280','S270', 'S271', 'S281', 'S290', 'S260']

#MNH
#Subjects = ['S312','S078', 'S069', 'S104','S088', 'S072']

#ONH
#Subjects = ['S308', 'S105', 'S291', 'S310']

#age = [18,19,19,19,19,19,20,20,20,20,21,21,21,21, 59, 61, 66, 70]

age = [18,19,19,19,19,19,20,20,20,20,21,21,21,21,26,27,28,30,33,35,35,
       37,38,39,47,49,51,52,54,55,59,60,61,66,70,71]

#Above 35 years
#Subjects = ['S312','S078', 'S069', 'S104','S088', 'S072','S308', 'S105', 'S291', 'S310']



# Subjects = ['S072','S078','S088','S259', 'S268', 'S269',
#             'S270', 'S271', 'S273', 'S274','S277','S279','S282',
#             'S285', 'S290' ,'S281','S291',
#             'S308','S310']

# age = [55, 47, 52, 20, 19, 19, 21, 21, 18, 19, 20, 20, 19,
#          19, 21, 21, 66, 59, 70]

A_epochs = []
A_evkd = []

#%% Load data

for subject in Subjects:
    print('Loading ' + subject)
    with open(os.path.join(data_loc,subject+'_Binding_0.4.pickle'),"rb") as file:
        [t, t_full, conds_save, epochs_save,evkd_save] = pickle.load(file)
    A_epochs.append(epochs_save)
    A_evkd.append(evkd_save)
#pickle.loads(open("C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Analysis/Pickles/S104_Binding.pickle", "rb").read())

# for subj in Subjects_YNH:
#     epochs_file = data_loc + subj + '_Epochs_Binding.fif'
#     evoked_file = data_loc + subj + '_Evoked_Binding.fif'
#     epochs = read_epochs(epochs_file)
#     evkd = read_evokeds(evoked_file)

#%% 32 Channel responses

evkd_12 = []
evkd_20 = []
for sub in range(len(Subjects)):
    evkd_12.append(A_evkd[sub][6].data)
    evkd_20.append(A_evkd[sub][7].data)

#%% Get evoked responses

A_evkd_cz = np.zeros((len(t),len(Subjects),len(conds_save)-2))

for sub in range(len(Subjects)):
    for cond in range(len(conds_save)-2):
        A_evkd_cz[:,sub,cond] = A_epochs[sub][cond].mean(axis=0)

#%% Plot Average response across Subjects (unNormalized)

#Conds:
#   0 = 12 Onset
#   1 = 20 Onset
#   3 = 12AB
#   4 = 12BA
#   5 = 20AB
#   6 = 20BA

cond_bind = ['12 Onset', '20 Onset', '12AB', '12BA', '20AB', '20BA', '12 all','20 all']

conds_comp = [[0,1], [2,4], [3,5]]
labels = ['Onset', 'Incoherent to Coherent', 'Coherent to Incoherent']

fig,ax = plt.subplots(3,1,sharex=True)
#fig.set_size_inches(20,10)
#plt.rcParams.update({'font.size': 26})


for jj in range(3):
    cnd1 = conds_comp[jj][0]
    cnd2 = conds_comp[jj][1]

    onset12_mean = ((A_evkd_cz[:,:,cnd1]).mean(axis=1))*1e6
    onset12_sem = (A_evkd_cz[:,:,cnd1]*1e6).std(axis=1) / np.sqrt(A_evkd_cz.shape[1])

    ax[jj].plot(t,onset12_mean,label='12')
    ax[jj].fill_between(t,onset12_mean - onset12_sem, onset12_mean + onset12_sem,alpha=0.5)

    onset20_mean = (A_evkd_cz[:,:,cnd2]*1e6).mean(axis=1)
    onset20_sem = (A_evkd_cz[:,:,cnd2]*1e6).std(axis=1) / np.sqrt(A_evkd_cz.shape[1])

    ax[jj].plot(t,onset20_mean,label='20')
    ax[jj].fill_between(t,onset20_mean - onset20_sem, onset20_mean + onset20_sem,alpha=0.5)

    ax[jj].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    #ax[jj].set_title(labels[jj])
    ax[jj].tick_params(labelsize=12)


ax[0].legend(fontsize=12)
ax[0].set_title('Onset', loc='center', fontsize=10)
ax[1].set_title('Incoherent to Coherent', loc='center', fontsize=10)
ax[2].set_title('Coherent to Incoherent', loc='center', fontsize=10)
ax[2].set_xlabel('Time (s)',fontsize=14)
ax[1].set_ylabel('Amplitude (\u03bcV)',fontsize=14)
ax[2].set_xlim([-0.050,1])
ax[2].set_xticks([0,0.2,0.4,0.6,0.8,1.0])
#ax[2].set_ylabel('$\mu$V')

fig.suptitle('Binding Across Subjects (N=' + str(len(Subjects)) + ') - 0.4 to 40 Hz')
plt.rcParams["figure.figsize"] = (6.5,5)
plt.show()


plt.savefig(save_loc+'AllSubj_12vs20_baselined_0.4.png', dpi=300)

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

cond_bind = ['12 Onset', '20 Onset', '12AB', '12BA', '20AB', '20BA', '12 all','20 all']

conds_comp = [[0,1], [2,4], [3,5]]
labels = ['Onset', 'Incoherent to Coherent', 'Coherent to Incoherent']

young = np.array(age) <= 35
old = np.array(age) > 35

onset20_mean_ynh = (A_evkd_cz[:,young,cnd1]*1e6).mean(axis=1)
onset20_sem_ynh = (A_evkd_cz[:,young,cnd1]*1e6).std(axis=1) / np.sqrt(A_evkd_cz[:,young,cnd1].shape[1])
onset20_peak_ynh = onset20_mean_ynh [t > 0]
print(max(onset20_peak_ynh))

onset20_mean_onh = (A_evkd_cz[:,old,cnd2]*1e6).mean(axis=1)
onset20_sem_onh = (A_evkd_cz[:,old,cnd2]*1e6).std(axis=1) / np.sqrt(A_evkd_cz[:,old,cnd2].shape[1])
onset20_peak_onh = onset20_mean_onh [t > 0]
print(max(onset20_peak_onh))

name1 = ['Below 35y']
name2=['Above 35y']
fig, ax = plt.subplots()
ax.bar(name1,onset20_mean_ynh,yerr=(onset20_sem_ynh))
ax.bar(name2,onset20_mean_onh,yerr=(onset20_sem_onh))
plt.show()


# age_class = []
# for a in age:
#     if np.isnan(a):
#         age_class.append(0) #i think its a young person

#     elif (a <= 35):
#         age_class.append(0)
#     elif((a >35)): #and (a <=55)):
#         age_class.append(1)
#     else:
#           age_class.append(2)

# YNH = np.array[(age) >= 18 and <=35]
# MNH = np.array(age) >= 36 and <=55
# ONH = np.array(age) > 55


fig,ax = plt.subplots(3,1,sharex=True)

conds_comp = [[1,1], [4,4], [5,5]]
labels = ['Onset', 'Incoherent to Coherent 20', 'Coherent to Incoherent 20']

for jj in range(3):
    cnd1 = conds_comp[jj][0]
    cnd2 = conds_comp[jj][1]

    onset12_mean = (A_evkd_cz[:,young,cnd1]*1e6).mean(axis=1)
    onset12_sem = (A_evkd_cz[:,young,cnd1]*1e6).std(axis=1) / np.sqrt(A_evkd_cz[:,young,cnd1].shape[1])

    ax[jj].plot(t,onset12_mean,label='Young (N=21)',color='tab:green')
    ax[jj].fill_between(t,onset12_mean - onset12_sem, onset12_mean + onset12_sem,alpha=0.5, color='tab:green')

    onset20_mean = (A_evkd_cz[:,old,cnd2]*1e6).mean(axis=1)
    onset20_sem = (A_evkd_cz[:,old,cnd2]*1e6).std(axis=1) / np.sqrt(A_evkd_cz[:,old,cnd2].shape[1])

    ax[jj].plot(t,onset20_mean,label='Old (N=15)', color='tab:purple')
    ax[jj].fill_between(t,onset20_mean - onset20_sem, onset20_mean + onset20_sem,alpha=0.5,color='tab:purple')

    ax[jj].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax[jj].set_title(labels[jj])


ax[0].legend(prop={'size': 6})
ax[2].set_xlabel('Time (s)',fontsize=14)
ax[1].set_ylabel('Amplitude (\u03bcV)',fontsize=14)
ax[2].set_xlim([-0.050,1])
ax[2].set_xticks([0,0.2,0.4,0.6,0.8,1.0])
fig.suptitle('Binding Across Age (N=' + str(len(Subjects)) + ')')
#ax[2].set_ylabel('$\mu$V')
#fig.suptitle('Binding Across Subjects (N=27)')
plt.rcParams["figure.figsize"] = (6.5,5)
plt.show()

#ax[2].set_ylabel('$\mu$V')
plt.savefig(save_loc+'YoungvsOld_0.4.png', dpi=500)



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
ax[0].set_xlabel('Time (s)')


plt.show()


#%% Normalize Data to max of of abs of onset response

# for sub in range(len(Subjects)):
#      sub_norm = np.max(np.abs((A_epochs[sub][0].mean(axis=0) + A_epochs[sub][1].mean(axis=0)) / 2))
#      for cnd in range(len(cond_bind)):
#         A_epochs[sub][cnd] /= sub_norm

#%% Lets look at Full time

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
plt.title('Binding Across Age (N=' + str(len(Subjects)) + ')')

plt.savefig(save_loc+'All_12vs20_noBaseline_0.4.png', dpi=500)


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

#%% Extract Features from EEG


#Feature List
    #0: onset power (12,20)
    #1: A onset power 12
    #2: A onset power 20
    #3: B onset power 12
    #4: B onset power 20
    #5: onset mean (12,20)
    #6: A12 mean
    #7: A20 mean
    #8: B12 mean
    #9: B20 mean

# feat_labels = ['Onset', 'A12_On', 'A20_on', 'B12_on', 'B20_on',
#                'Onset_mean', 'A12_mean', 'A20_mean', 'B12_mean',
#                'B20_mean' ]

# features = np.zeros([len(Subjects), 10])

# onset_inds = [3, 5, 2, 4]

# # ta_1 = [0.075, 0.11, 0.16]
# # ta_2 = [1.4, 0.16, 0.3]

# # tb_1 = [0.1, 0.225, 0.325,0.45]
# # tb_2 = [0.225, 0.325, 0.45, 0.9]

# t_0 = np.where(t>=0.050)[0][0]
# t_2 = np.where(t>=0.400)[0][0]
# t_3 = np.where(t>=0.800)[0][0]
# t_e = np.where(t>=1)[0][0]

# for sub in range(len(Subjects)):

#     resp_on = (A_epochs[sub][0].mean(axis=0) + A_epochs[sub][1].mean(axis=0) ) / 2 # mean onset (12,20)
#     features[sub,0] = resp_on[t_0:t_2].var() / resp_on[t_0:t_e].var() # roughly the % var in first 't_2' seconds
#     features[sub,5] = resp_on[t_2:t_3].mean() # mean b/t t_2 - t_3

#     for on in range(4):
#         resp = A_epochs[sub][onset_inds[on]].mean(axis=0)
#         features[sub,on+1] = resp[t_0:t_2].var() / resp[t_0:t_e].var()
#         features[sub,on+6] = resp[t_2:t_3].mean() #/ resp_on[t_0:t_e].var()

# #%% waveform pca for feature selection

# t_0 = np.where(t >= 0)[0][0]
# t_1 = np.where(t > 1)[0][0]

# O_resps = np.zeros((len(Subjects),(t_1-t_0)))
# B_resps = np.zeros((len(Subjects),(t_1-t_0)*2))
# A_resps = np.zeros((len(Subjects),(t_1-t_0)*2))
# AB_resps = np.zeros((len(Subjects),(t_1-t_0)*4))

# for sub in range(len(Subjects)):
#     Bresp = A_epochs[sub][2][:,t_0:t_1].mean(axis=0)
#     Bresp2 = A_epochs[sub][4][:,t_0:t_1].mean(axis=0)

#     Aresp = A_epochs[sub][3][:,t_0:t_1].mean(axis=0)
#     Aresp2 = A_epochs[sub][5][:,t_0:t_1].mean(axis=0)

#     On_resp = (A_epochs[sub][0][:,t_0:t_1].mean(axis=0) + A_epochs[sub][1][:,t_0:t_1].mean(axis=0) ) / 2

#     O_resps[sub,:] = On_resp
#     A_resps[sub,:] = np.concatenate((Aresp,Aresp2))
#     B_resps[sub,:] = np.concatenate((Bresp,Bresp2))
#     AB_resps[sub,:] = np.concatenate((Aresp,Aresp2,Bresp,Bresp2))


# pca = PCA(n_components=3)

# #Aall_feature = pca.fit_transform(StandardScaler().fit_transform(A_resps))
# Aall_feature = pca.fit_transform(A_resps)
# Aall_expVar = pca.explained_variance_ratio_
# Aall_comp = pca.components_

# #Ball_feature = pca.fit_transform(StandardScaler().fit_transform(B_resps))
# Ball_feature = pca.fit_transform(B_resps)
# Ball_expVar = pca.explained_variance_ratio_
# Ball_comp = pca.components_

# ABall_feature = pca.fit_transform(StandardScaler().fit_transform(AB_resps))
# ABall_expVar = pca.explained_variance_ratio_
# ABall_comp = pca.components_

# #O_feature = pca.fit_transform(StandardScaler().fit_transform(O_resps))
# O_feature = pca.fit_transform(O_resps)
# O_expVar = pca.explained_variance_ratio_
# O_comp = pca.components_

# plt.figure()
# plt.scatter(O_feature[:,0],Ball_feature[:,0])

# plt.figure()
# plt.scatter(O_feature[:,0],Aall_feature[:,0])

# plt.figure()
# #plt.plot(t[t_0:t_1],O_comp[0,:],label='Onset',linewidth='2')
# plt.plot(np.concatenate((t[t_0:t_1],t[t_0:t_1]+1+1/4096)),Aall_comp[0,:], label= 'Incoherent', color='Grey')
# plt.plot(np.concatenate((t[t_0:t_1],t[t_0:t_1]+1+1/4096)),-Ball_comp[0,:], label ='Coherent', color='Black')
# plt.xticks([0,0.5,1.0, 1.5, 2.0])
# plt.yticks([0, 0.01, .02])
# plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
# plt.legend(loc=2)
# plt.ylabel('PCA Feature Weight')
# plt.xlabel('Time (sec)')
# #plt.title('EEG Feature From PCA')

# plt.savefig(os.path.join(fig_loc,'PCA_FeatWeights.svg'),format='svg')

# fig = plt.figure()
# #fig.set_size_inches(12,10)
# plt.rcParams.update({'font.size': 12})
# #plt.plot(t[t_0:t_1],O_comp[0,:],label='Onset',linewidth='2')
# plt.plot(t[t_0:t_1], Aall_comp[0,4097:], label= 'Incoherent',linewidth='2')
# plt.plot(t[t_0:t_1], -Ball_comp[0,4097:], label ='Coherent',linewidth='2')
# plt.legend(loc = 4)
# plt.ylabel('Weight')
# plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
# plt.xlabel('Time (s)')
# plt.title('EEG Feature Weights From PCA')
# plt.xticks([0,0.5,1.0])
# #plt.yticks([0, 0.02, 0.04])
# plt.yticks([0, 0.01, .02])




# #plt.plot(np.concatenate((t[t_0:t_1],t[t_0:t_1]+1+1/4096, t[t_0:t_1]+2+1/4096,t[t_0:t_1]+3+1/4096 )),
#  #        ABall_comp[0,:])

# plt.figure()
# plt.scatter(Aall_feature[:,0],Aall_feature[:,0]/ O_feature[:,0])

# ABall_feature = ABall_feature[:,0] #/ O_feature[:,0]
# Aall_feature = Aall_feature[:,0] #/ O_feature[:,0]
# Ball_feature = Ball_feature[:,0] #/ O_feature[:,0],
# O_feature = O_feature[:,0]