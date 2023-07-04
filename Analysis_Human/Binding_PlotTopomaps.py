# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 19:29:54 2023

@author: vmysorea
"""
import sys
sys.path.append('C:/Users/vmysorea/Documents/mne-python/')
import warnings
import mne
from matplotlib import pyplot as plt
from scipy.io import savemat
import numpy as np

plt.switch_backend('QT5Agg')  # Making the plots interactive (Scrollable)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# Defining the dimensions and quality of figures
plt.rcParams['figure.figsize'] = [4,4]
plt.rcParams['figure.dpi'] = 120

plt.rcParams.update({
    'ytick.labelsize': 'small',
    'xtick.labelsize': 'small',
    'axes.labelsize': 'small',
    'axes.titlesize': 'medium',
    'grid.color': '0.75',
    'grid.linestyle': ':',
})

#%%Picking up saved epochs
save_loc='C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Human_Analysis/Figures/'
save_epochs_loc = 'D:/PhD/Data/Epochs-fif/'

subjlist = ['S312','S347','S340','S078','S069','S088','S072','S308','S344','S105','S291','S310','S339']
 # ,'S312','S347','S340','S078','S069','S088',
 # 'S072','S308','S344','S105','S291','S310','S339'           # 
            # 'S274','S282','S285','S259','S277','S279','S280',
            # 'S270','S271','S281','S290','S284','S305','S303','S288','S260','S341','S312','S347','S340'] #'S337' Missing EXGs , 'S069', 'S104' - Something wrong here?
# 'S273','S277','S279','S280','S270','S271','S281','S290','S284',
               # 'S305','S303'
            # ,'S078','S088','S072','S308','S344',
            # 'S105','S345','S291','S310','S339'
            
#Done: 
condlist = [1, 2]  # List of conditions- Coherence of 12 and 20 tones
condnames = ['12', '20']
evokeds = []
# mapping={}
# evokeds_1 = []
# evokeds_2 = []

for subj in subjlist:
    evoked= mne.read_evokeds(save_epochs_loc + subj +'_Binding_evoked20_0.4-ave.fif',baseline=(-0.3,0))
    # evoked = epochs.average()
    # evoked.save(save_epochs_loc + subj +'_Binding_evoked-ave.fif',overwrite=True, verbose='INFO')
    evokeds+=[evoked,]

for c in range(len(subjlist)):
    evokeds_all = mne.combine_evoked(evokeds[c], weights='equal')

# for subj in subjlist:
#     x = evokeds.get_data()
#     x_mean = x.mean(axis=0)
#     x_sem = x.std(axis=0) / np.sqrt(x.shape[0])
#     t=epochs.times
#     mat_ids = dict(evoked=x_mean, evoked_sem = x_sem, fs=4096, t=epochs.times)
#     savemat(save_loc + 'allevoked_N24_allchans.mat', mat_ids)
    
# evokeds.plot(picks=['A31', 'A32'])
times=(0.2, 0.6,1.2,1.6,2.2,2.6,3.1,3.6,4.2,4.6)
# plt.rcParams["figure.figsize"] = (6,7)
mask=evokeds_all.data>1e-6
fig = evokeds_all.plot_topomap(times=times,average=0.3,
                               contours=15, res=128, size=1, 
                               colorbar=False,extrapolate='head',
                               mask=mask)
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.88)
fig.tight_layout()
fig.suptitle('Above 35 years', fontsize=14)

plt.savefig(save_loc + 'Evoked20_topo_Above35.png', dpi=500)

evokd = evokeds_all.plot_topomap(times=times,average=0.3,contours=15, res=128, size=4, colorbar=True)

evokeds_all.plot_joint(times, title='Evoked Response for Binding Stimulus (20 tone)')
