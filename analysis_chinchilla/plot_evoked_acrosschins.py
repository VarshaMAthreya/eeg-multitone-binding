# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 13:42:37 2023

@author: vmysorea
"""

import sys
sys.path.append('C:/Users/vmysorea/Documents/mne-python/')
import warnings
from matplotlib import pyplot as plt
from scipy import io
import numpy as np
from scipy.stats import sem

plt.switch_backend('QT5Agg')  # Making the plots interactive (Scrollable)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# Defining the dimensions and quality of figures
plt.rcParams["figure.figsize"] = (5.5,5)
plt.rcParams['figure.dpi'] = 120
#%%Setting up stuff

save_loc_fig='C:/Users/vmysorea/Desktop/PhD/Conferences/ARO 2024/Chin_miniEEG/FIgures/'
save_loc_mat='D:/PhD/Data/Chin_Data/AnalyzedBinding_matfiles/'

####
subjlist_1 = ['Q414']
evk_sedawake_all = np.zeros((len(subjlist_1),6145))

for subj in range(len(subjlist_1)):
    sub = subjlist_1 [subj]
    dat = io.loadmat(save_loc_mat + sub + '_BindingSedated.mat', squeeze_me=True)
    dat.keys()
    x = (dat['ep_cap']).mean(axis=0)
    t=dat['t']
    evk_sedawake_all[subj,:] = x
    
evk_sedawake = evk_sedawake_all.mean(axis=0)
sem_sedawake = sem(evk_sedawake_all)

#####
subjlist_2 = ['Q414', 'Q417']
evk_awake_all = np.zeros((len(subjlist_2),6145))

for subj in range(len(subjlist_2)):
    sub = subjlist_2 [subj]
    dat = io.loadmat(save_loc_mat + sub + '_BindingAwake.mat', squeeze_me=True)
    dat.keys()
    y = (dat['ep_cap']).mean(axis=0)
    t1=dat['t']
    evk_awake_all[subj,:] = y
    
evk_awake = evk_awake_all.mean(axis=0)
sem_awake = sem(evk_awake_all)

####
subjlist_3 = ['Q422', 'Q428']
evk_sed_all = np.zeros((len(subjlist_3),6145))

for subj in range(len(subjlist_3)):
    sub = subjlist_3 [subj]
    dat = io.loadmat(save_loc_mat + sub + '_BindingLightSedation.mat', squeeze_me=True)
    dat.keys()
    z = (dat['ep_cap']).mean(axis=0)
    t2=dat['t']
    evk_sed_all[subj,:] = z
    
evk_sed = evk_sed_all.mean(axis=0)
sem_sed = sem(evk_sed_all)

t3=t[t<0.7]
evk_sedawake1=evk_sedawake[t<0.7]
sem_sedawake1 = sem_sedawake[t<0.7]
t4=t1[t1<0.7]
evk_awake1 =evk_awake[t1<0.7]
sem_awake1 = sem_awake[t<0.7]
t5=t2[t2<0.7]
evk_sed1=evk_sed[t<0.7]
sem_sed1=sem_sed [t<0.7]

fig, ax = plt.subplots(constrained_layout=True)
# plt.errorbar(t3, evk_awake1*1e5, yerr=sem_awake1*1e5, label = 'Awake (N=2)', color='purple', linewidth=2, ecolor='thistle')
plt.errorbar(t3, evk_sedawake1*1e5, yerr=sem_sedawake1*1e5,  label = 'Anesthetized (N=2)', color='green', linewidth=2, ecolor='darkseagreen')
plt.errorbar(t3, evk_sed1*1e5, yerr=sem_sed1*1e5,  label = 'Light Sedation (N=2)', color='darkblue', linewidth=2, ecolor='lightsteelblue')
# plt.errorbar(t, b, yerr=s, label = 'Above 35 y (N=' + str(len(subjlist_o)) + ')', color='purple', linewidth=2, ecolor='thistle')
plt.title('Effects of Sedation on Onset Responses')
plt.xlabel('Time(s)',fontsize=20)
plt.ylabel('Amplitude(\u03bcV)',fontsize=20)
# plt.tight_layout()
plt.rcParams["figure.figsize"] = (6.5,5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper right',fontsize='x-large')
plt.show()

plt.savefig(save_loc_fig + 'EffectsofAnesthesia_woAwake_Binding', dpi=500, height=5, width=6)

