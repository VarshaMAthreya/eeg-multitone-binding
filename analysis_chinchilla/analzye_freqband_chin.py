# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 18:16:11 2023

@author: vmysorea
"""
import sys
sys.path.append('C:/Users/vmysorea/Documents/mne-python/')
import warnings
from matplotlib import pyplot as plt
from scipy import io
import numpy as np
from scipy.stats import sem
from scipy.signal import spectrogram
from scipy import signal
import math
from mne.viz import centers_to_edges
from mne.baseline import rescale

plt.switch_backend('QT5Agg')  # Making the plots interactive (Scrollable)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# Defining the dimensions and quality of figures
plt.rcParams["figure.figsize"] = (5.5,5)
plt.rcParams['figure.dpi'] = 120
#%%Setting up stuff

save_loc_fig='C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Chin_Analysis/Freq_Band/Figures/'
save_loc_mat='C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Chin_Analysis/Freq_Band/'

subjlist = ['Q419']

power=[]

theta_all = np.zeros((len(subjlist),8,4917))
alpha_all = np.zeros((len(subjlist), 8,4917))
beta_all = np.zeros((len(subjlist),46,4917))
lowgamma_all = np.zeros((len(subjlist),20,4917))
highgamma_all = np.zeros((len(subjlist),25,4917))

for subj in range(len(subjlist)):
    sub = subjlist [subj]
    dat = io.loadmat(save_loc_mat + sub + '_ABR_freqbands_anesthetized.mat', squeeze_me=True)
    dat.keys()
    a = (dat['theta']).mean(axis=0)
    b = (dat ['alpha']).mean(axis=0)
    c = (dat['beta']).mean(axis=0)
    d = (dat['low_gamma']).mean(axis=0)
    e = (dat['high_gamma']).mean(axis=0)
    picks=dat['picks']
    t=dat['t']
    theta_all[subj,:] = a
    alpha_all[subj,:] = b
    beta_all[subj,:] = c
    lowgamma_all[subj,:] = d 
    highgamma_all[subj,:] = e

theta_mean = (theta_all.mean(axis=0)) ### Averaging across channels and across subjects 
alpha_mean = (alpha_all.mean(axis=0))
beta_mean = (beta_all.mean(axis=0))
lowgamma_mean = (lowgamma_all.mean(axis=0))
highgamma_mean = (highgamma_all.mean(axis=0))

# power += [theta_mean, alpha_mean, beta_mean, lowgamma_mean, highgamma_mean]

theta_freqs = np.arange(4., 8., 0.5) 
alpha_freqs = np.arange(8., 12., 0.5) 
beta_freqs = np.arange(12., 35., 0.5)
lowgamma_freqs = np.arange(35., 55., 1.) 
highgamma_freqs= np.arange(55., 80., 1.)

# freqs = [theta_freqs, alpha_freqs,beta_freqs,low_gamma_freqs,high_gamma_freqs]

# Baseline the output

#beta, alpha:
vmin=-4*1e-8
vmax=6*1e-8

#theta: 
vmin=-1*1e-8
vmax=1.5*1e-8 

#low,high gamma:
vmin=-4*1e-9
vmax=3*1e-9

rescale(theta_mean, t, (-0.3, 0.0), mode="mean", copy=False)

fig, ax = plt.subplots()
x, y = centers_to_edges(t, theta_freqs)
mesh = ax.pcolormesh(x, y, theta_mean, cmap="RdBu_r")
ax.set_title("theta power - ABR- Anesthetized")
ax.set(ylim=theta_freqs[[0, -1]], xlabel="Time (s)")
fig.colorbar(mesh)
plt.tight_layout()
plt.show()

plt.savefig(save_loc_fig + 'Anes_ABR_theta_All.png', dpi=500)


# power = [theta_mean, alpha_mean, beta_mean, lowgamma_mean, highgamma_mean]


# # Baseline the output

# for a in range(len(power)):
#     rescale(power[0], t, (-0.3, 0.0), mode="mean", copy=False)
    

# fig, ax = plt.subplots(5, 1, figsize = (15,5), sharex=True)
# # for b in range(len(freqs)):
    
    
# x, y = centers_to_edges(t, theta_freqs)
# mesh = ax.pcolormesh(x, y, theta_mean, cmap="winter")
# ax.set_title("Alpha power - Across awake chins")
# ax.set(ylim=freqs[[0, -1]], xlabel="Time (ms)")
# fig.colorbar(mesh)
# plt.tight_layout()

# plt.show()


#%% ###Option 4 - Working now 
# power20_all = np.array(power20_all.mean(axis=0))
# power20_all = np.array(power20_all.mean(axis=0))

fs=4096
interval = int(fs/5)
overlap=int(interval*0.95)

f4,t4,Sxx4 = signal.spectrogram (power20_all,fs=fs,nperseg=interval,
                         noverlap=overlap)
plt.pcolormesh(t4,f4,10*np.log(Sxx4), cmap='inferno')
plt.colorbar()
plt.title('ONH - All Gamma')
plt.ylim([30,80])
# plt.xlim([-0.3,5.1])
plt.show()

