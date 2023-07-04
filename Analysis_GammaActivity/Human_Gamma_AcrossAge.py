# -*- coding: utf-8 -*-
"""
Created on Thu May  4 00:48:52 2023

@author: vmysorea
"""
import sys
sys.path.append('C:/Users/vmysorea/Documents/mne-python/')
import warnings
from matplotlib import pyplot as plt
from scipy.io import savemat
from scipy import io
import numpy as np
from scipy.stats import sem
from scipy.signal import spectrogram
from scipy import signal
import math
from mne.baseline import rescale
from mne.viz import centers_to_edges

plt.switch_backend('QT5Agg')  # Making the plots interactive (Scrollable)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# Defining the dimensions and quality of figures
plt.rcParams["figure.figsize"] = (5.5,5)
plt.rcParams['figure.dpi'] = 120
#%%Setting up stuff
save_loc='C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Human_Analysis/Figures/'
epochs_loc = 'D:/PhD/Data/Epochs-fif/'
save_mat_loc ='C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Human_Analysis/Matfiles(Avg)/Gamma_Matfiles/'

### Haven't considered S273 and S345
subjlist = ['S268', 'S269', 'S274', 'S282', 'S285',
            'S277', 'S279', 'S280','S259','S270', 'S271', 
            'S281','S290', 'S284', 'S305','S303','S288','S260',
            'S352', 'S341', 'S312', 'S347', 'S340','S078','S069',
            'S088','S342','S072','S308','S344','S291','S310','S339']

subjlist_y = ['S273','S268', 'S269', 'S274', 'S282', 'S285',
            'S277', 'S279', 'S280','S259','S270', 'S271', 
            'S281','S290', 'S284', 'S305','S303','S288','S260',
            'S352', 'S341']

subjlist_o = ['S312', 'S347', 'S340','S078',
              'S088','S342','S072','S308','S344',
              'S291','S310','S339']

#S104 and S345 excluded (weird data) - S337 no EXGs
#%% Loading files for 12 and 20 condition for all subjects - No age separation 
# evokeds_12 = []
# evokeds_20 =[]
# power12 = []
# power20_all =[]
# power12_low =[]
# power20_low=[]
# power12_high=[]
# power20_high=[]

power12_all = np.zeros((len(subjlist_o),25,22120))
power20_all = np.zeros((len(subjlist_o),25,22120))
power12_low = np.zeros((len(subjlist_o),15,22120))
power20_low = np.zeros((len(subjlist_o),15,22120))
power12_high = np.zeros((len(subjlist_o),15,22120))
power20_high = np.zeros((len(subjlist_o),15,22120))

for subj in range(len(subjlist_o)):
    sub = subjlist_o [subj]
    dat = io.loadmat(save_mat_loc + sub + '_A32_gamma.mat', squeeze_me=True)
    dat.keys()
    a = (dat['power12_all'])
    b = (dat ['power20_all'])
    c = (dat['power12_low'])
    d = (dat['power20_low'])
    e = (dat['power12_high'])
    f = (dat['power20_high'])
    freqs= dat['freqs']
    low_freqs=dat['low_freqs']
    high_freqs=dat['high_freqs']
    picks=dat['picks']
    n_cycles=dat['n_cycles']
    n_cycles1=dat['n_cycles1']
    n_cycles2=dat['n_cycles2']
    t=dat['t']
    power12_all[subj,:] = a
    power20_all[subj,:] = b
    power12_low[subj,:] = c
    power20_low[subj,:] = d 
    power12_high[subj,:] =e
    power20_high[subj,:] = f

power20_low=np.array(power20_low.mean(axis=0))
power20_low=np.array(power20_low.mean(axis=0))
power12_all=power12_all.T

# nFFT=int(11060)
# fs=int(4096)
# # interval = int(fs)
# noverlap = int(nFFT-1)
# window=np.hanning(22120)

# # power12_all = np.array(power12_all.mean(axis=0))
# # power12_all=power12_all.T
# ##Option 1

# Pxx,low_freqs,bins,im=plt.specgram(power20_low, nFFT=nFFT,fs=fs, noverlap=noverlap)
# # plt.plot(t,(power12_all.mean(axis=0))*1e10)
# # plt.pcolormesh(t, f, Sxx, shading='auto',cmap = 'inferno')
# # plt.ylim(30,80)
# plt.show()

# # plt.plot(t,a.mean(axis=0))
# # plt.show()

# ###Option 2 
# nFFT = 44240
# fs=4096
# noverlap=nFFT*0.5

# freqs,times,Sxx = signal.spectrogram(power20_low.mean(axis=0), fs=fs,noverlap=0.95)
# Sxx=Sxx.mean(axis=0)
# Sxx=Sxx.T
# plt.pcolormesh(times, freqs, Sxx.T, shading='auto',cmap = 'inferno')
# plt.ylim(30,50)
# plt.show()

# t=t/2
# Sxx=Sxx.mean(axis=0)
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# fig, ax = plt.subplots(figsize=(5, 6))
# image = ax.pcolormesh(t, freqs, 10+ np.log10(Sxx), cmap='jet')
# ax.axis("image")
# ax.set_ylabel('Frequency [Hz]')
# ax.set_xlabel('Time [s]')
# divider = make_axes_locatable(ax)
# ax_cb = divider.new_horizontal(size="8%", pad=0.05)
# fig.add_axes(ax_cb)
# plt.colorbar(image, cax=ax_cb)
# plt.show()

# ###Option 3 
# power20_low=(power20_low.mean(axis=0))

# f,t,Sxx=signal.spectrogram(power20_low,fs=4096)
# plt.pcolormesh(t,f,Sxx,cmap='jet')
# plt.ylim(30,50)
# plt.xticks([-0.3,0,1,2,3,4])
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

###Option 5 -- Working in chins 

# Baseline the output

power20_high = (power20_high.mean(axis=0))
# rescale(power20_high, t, (-0.3, 0.0), mode="mean", copy=False)

vmin=-1*1e-10
vmax=2.5*1e-10
#

fig, ax = plt.subplots()
x, y = centers_to_edges(t, high_freqs)
mesh = ax.pcolormesh(x, y, power20_high, cmap='RdBu_r',vmin=vmin, vmax=vmax)
ax.set_title("High Gamma NB (65-80 Hz) - ONH (N=" + str(len(subjlist_o)) + ")", y=1.03)
ax.set(ylim=high_freqs[[0, -1]], xlabel="Time (s)")
fig.colorbar(mesh)
plt.tight_layout()
plt.show()

plt.savefig(save_loc + 'ONH_highGamma_NB.png', dpi=500)
