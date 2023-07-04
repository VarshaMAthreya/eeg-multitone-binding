# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 23:04:33 2023

@author: vmysorea
"""
import sys
import os
sys.path.append('C:/Users/vmysorea/Documents/mne-python/')
sys.path.append('C:/Users/vmysorea/Documents/ANLffr-master/')
from scipy import io
from scipy.io import savemat
import warnings
import mne
import numpy as np
from anlffr.helper import biosemi2mne as bs
from matplotlib import pyplot as plt
import fnmatch
from scipy import signal 
from scipy.signal import hilbert
import resampy
from scipy.stats import sem


plt.switch_backend('QT5Agg')  # Making the plots interactive (Scrollable)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# Defining the dimensions and quality of figures
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 120

#%% Setting locs and loading data

froot = 'D:/PhD/Data/Chin_Data/'  # file location
save_loc_mat='C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Chin_Analysis/EFR/matfiles/'
save_loc_fig='C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Chin_Analysis/EFR/Figures/'

#%%% Stim envelope extraction for the coherent conditions only
dat = io.loadmat('D:/PhD/Heinz Lab/Varsha_Binding/Binding_Biosemi/ChinBindingStims_12Only', squeeze_me=True)
dat.keys()
x = dat['A_stims']  #All coherent and incoherent stims 
trig = dat['trigs']
fs=48828
sfreq = 4096

# t = np.arange(0,x[0].size/fs,1/fs)

#The entire duration resampling
for c in range(len(x)):
    x[c,]=resampy.resample(x[c],fs, sfreq)          #Resampling to match the EEG Fs, for accurate cross-correlation 

t = np.arange(0,x[0].size/sfreq,1/sfreq)

x_avg = x.mean(axis=0)      #Need later for plotting only

###Extracting incoherent and coherent parts of the stimulus for all trials
tincoh1 = t[(t>=0) & (t<1)]
x1=x[0][(t>=0) & (t<1)] #Single trial
tcoh1 = t[(t>=1) & (t<2)]
# x2=x[0][(t>=1) & (t<=2)] #Single trial
tincoh2 = t[(t>=2) & (t<3)]
# x3=x[0][(t>=2) & (t<=3)] #Single trial
tcoh2 = t[(t>=3) & (t<4)]
# x4=x[0][(t>=3) & (t<=4)] #Single trial 
tincoh3 = t[(t>=4)]
# x5=x[0][(t>=4) & (t<=5)] #Single trial

coh1=np.zeros((len(x),len(tcoh1)))
coh2=np.zeros((len(x),len(tcoh2)))
incoh1=np.zeros((len(x),len(tincoh1)))
incoh2=np.zeros((len(x),len(tincoh2)))
incoh3=np.zeros((len(x),len(tincoh3)))

for c in range(len(x)):
    coh1[c,]=x[c][(t>=1) & (t<2)]
    coh2[c,]=x[c][(t>=3) & (t<4)]
    incoh1[c,]=x[c][(t>=0) & (t<1)]
    incoh2[c,]=x[c][(t>=2) & (t<3)]
    incoh3[c,]=x[c][(t>=4)]
    
# coh1=coh1.mean(axis=0)
# coh2=coh2.mean(axis=0)

# plt.plot(tcoh1,coh1,linewidth=2)
# plt.plot(tcoh2,coh2,linewidth=2)

# plt.plot(tcoh1,x1,linewidth=2)
# plt.plot(tcoh2,x2,linewidth=2)

###Hilbert transform to get envelope of stimulus for coherent and incoherent components
sig1=np.zeros((len(x), len(tincoh1)))
amp1=np.zeros((len(x), len(tincoh1)))
sig2=np.zeros((len(x), len(tcoh1)))
amp2=np.zeros((len(x), len(tcoh1)))
sig3=np.zeros((len(x), len(tincoh2)))
amp3=np.zeros((len(x), len(tincoh2)))
sig4=np.zeros((len(x), len(tcoh2)))
amp4=np.zeros((len(x), len(tcoh2)))
sig5=np.zeros((len(x), len(tincoh3)))
amp5=np.zeros((len(x), len(tincoh3)))

for c in range(len(x)):
    sig1[c,]=hilbert(incoh1[c])
    # sig[c,]=(hilbert(incoh1[c]))#To get that nice envelope, need the imaginary part of the signal which was getting thrown out with just hilbert(x)
    amp1[c,]=np.abs(sig1[c])
    # amp[c,]=np.abs(sig[c])
    sig2[c,]=hilbert(coh1[c])
    amp2[c,]=np.abs(sig2[c])
    sig3[c,]=hilbert(incoh2[c])
    amp3[c,]=np.abs(sig3[c])
    sig4[c,]=hilbert(coh2[c])
    amp4[c,]=np.abs(sig4[c])
    sig5[c,]=hilbert(incoh3[c])
    amp5[c,]=np.abs(sig5[c])
    

plt.plot(tcoh1,sig2[0], label = 'Analytic signal')
# plt.plot(tcoh1,sig[0], label = 'Analytic signal')
plt.plot(tcoh1,amp2[0], label = 'Abs analytic')
# plt.plot(tincoh1,amp[0], label = 'Amp analytic')
plt.plot(tcoh1,coh1[0], label='Original')
plt.legend()
plt.show()

mat_ids_stim = dict(amp1=amp1,amp2=amp2,amp3=amp3,amp4=amp4,amp5=amp5)
savemat(save_loc_mat + 'StimEnvelope_12Only.mat', mat_ids_stim)   

stim = io.loadmat(save_loc_mat + 'StimEnvelope_12Only.mat', squeeze_me=(True))
stim.keys()
amp1 = stim['amp1'] 
amp2 = stim['amp2']
amp3 = stim['amp3']
amp4 = stim['amp4']
amp5 = stim['amp5']

#For full 5 second interval
sig=np.zeros((len(x), len(t)))
# sig_i=np.zeros((len(x), len(t)))
amp=np.zeros((len(x), len(t)))
# amp_i=np.zeros((len(x), len(t)))

for c in range(len(x)):
    sig[c,]=np.imag(hilbert(x[c]))
    #sig_i[c,]=np.imag(hilbert(x[c]))
    amp[c,]=np.abs(sig[c])
    #amp_i[c,]=np.abs(sig_i[c])

# plt.plot(t,sig[0], label = 'Real')
# plt.plot(t,amp[0], label = 'Amp real')
# plt.plot(t,sig_i[0], label = 'Imag')
# plt.plot(t,amp_i[0], label = 'Amp imag')
# plt.plot(t,x[0], label='Original')
# plt.legend()

##FFT
# x1f = np.fft.fft(coh1,axis=0)
# N1 = x1f.size
# f1 = np.fft.fftfreq(N1, d=1/fs) # get frequency vector
# #Just look at postivie frequencies since fft is symmetric for real signals
# x1f = x1f[f1>=0]
# f1 = f1[f1>=0]
# plt.plot(f1,abs(x1f), label='Coherence 1')
# plt.xlabel('Frequency (Hz)',fontsize=14)
# plt.ylabel('Magnitude',fontsize=14)
# #ax[0].set_xlim([0,1000])
# plt.title('FFT',fontweight='bold',fontsize=14)

# x2f = np.fft.fft(coh2,axis=0)
# N2 = x2f.size
# f2 = np.fft.fftfreq(N2, d=1/fs) # get frequency vector
# #Just look at postivie frequencies since fft is symmetric for real signals
# x2f = x2f[f2>=0]
# f2 = f2[f2>=0]
# plt.plot(f2,abs(x2f),label='Coherence 2')
# plt.xlabel('Frequency (Hz)',fontsize=14)
# plt.ylabel('Magnitude',fontsize=14)
# #ax[0].set_xlim([0,1000])
# plt.legend()
# plt.title('FFT',fontweight='bold',fontsize=14)
# plt.show()

#%%Evoked Response -- Get EFR 
    
subjlist = ['Q419']  # Load subject folder
# condlist = [1, 2] #Coherence of 12 and 8 tones (Reversed here!!!!!!)
# condnames = ['12', '8']

for subj in subjlist:
    evokeds = []
    # Load data and read event channel
    fpath = froot + subj + '/'
    bdfs = fnmatch.filter(os.listdir(fpath), subj +
                          '_ACC_F103_20Percent_500ms_AWAKE*.bdf')

    # Load data and read event channel
    rawlist = []
    evelist = []

    for k, rawname in enumerate(bdfs):
        rawtemp, evestemp = bs.importbdf(fpath + rawname, verbose='DEBUG',
                                         refchans=['EXG1', 'EXG2'])
        rawtemp.set_channel_types({'EXG3':'eeg'})           #Mastoid
        rawtemp.set_channel_types({'EXG4':'eeg'})           #Vertex
        rawtemp.set_channel_types({'EXG5':'eeg'})           #Ground
        rawlist += [rawtemp, ]
        evelist += [evestemp, ]
    raw, eves = mne.concatenate_raws(rawlist, events_list=evelist)
    # raw, eves = raw.resample(8192, events=eves)
    raw.info['bads'].append('EXG6') 
    raw.info['bads'].append('EXG7')
    raw.info['bads'].append('EXG8')
    raw.info['bads'].append('A26')
    raw.info['bads'].append('A27')
    raw.info['bads'].append('A12')
    
    raw.set_annotations(None)
    
    #raw.set_eeg_reference('average', projection =True)

    # To check and mark bad channels
    raw.plot(duration=25.0, n_channels=41, scalings=dict(eeg=200e-6))
    
# %% Filtering

    raw.filter(1.,25.)
    raw.info
    sfreq=raw.info['sfreq']
    raw.plot(duration=25.0, n_channels=41, scalings=dict(eeg=200e-6))

#%% Plotting Onset responses for all conditions included

    epochs = mne.Epochs(raw, eves, event_id=[1], baseline=None, tmin=0., tmax=5., reject=dict(eeg=200e-6), 
                        reject_by_annotation=False, preload=False, proj=False)
    all_channels = (np.arange(0,31))
    epochs = epochs.get_data()[:, all_channels,:]
    ep_mean = epochs.mean(axis=1)
    
    plt.plot(t,ep_mean, label='8')
    plt.show()
    
    
    
    epochs_down = mne.Epochs(raw, eves, event_id=[1,2], baseline=(-0.3,0), tmin=-0.3, tmax=1.2, reject=dict(eeg=200e-6), 
                        reject_by_annotation=False, preload=False, proj=False)
    all_channels = (np.arange(0,31))
    picks = (1,2,17,6,22,21)
    # epoch_12 = epochs_12.get_data()[:, all_channels,:]
    # ep_mean = epoch_12.mean(axis=1)
    
    t=epochs_down.times
    epochs_up = mne.Epochs(raw, eves, event_id=[2,4], baseline=(-0.3,0), tmin=-0.3, tmax=1.2, reject=dict(eeg=200e-6), 
                             reject_by_annotation=False, preload=False, proj=False)
    # epoch_up = epochs_up.get_data()[:,all_channels,:]
    # epochs_down = epochs_down.get_data()[:,all_channels,:]
    ep_up_mean = ((epochs_up.get_data()[:,all_channels,:]).mean(axis=1))*1e6
    ep_down_mean =((epochs_down.get_data()[:,all_channels,:]).mean(axis=1))*1e6
   
    #plt.plot(t,ep_up_mean.mean(axis=0), label='Up')
    plt.plot(t,ep_down_mean.mean(axis=0), label='Down')
    # plt.plot(t,epoch_8, label='8')
    plt.title(subj + ' - Awake - A8')
    # plt.xlim(-0.2,0.4)
    # plt.ylim(-5,15)
    plt.legend()
    plt.show()
    
    
    
    plt.savefig(save_loc_fig + subj + '_Awake_5sec_A8', dpi=500)

    
    
    mat_ids = dict(ep_up_mean=ep_up_mean, ep_down_mean=ep_down_mean, sfreq=sfreq)
    io.savemat(save_loc_mat + subj + '_ACC_TrialEpochs.mat', mat_ids)

#Load already analyzed mat file for this chin 
# y_all=[]
subjlist = ['Q419']  # Load subject folder
for subj in subjlist:    
    dat1 = io.loadmat(save_loc_mat + subj + '_ACC_TrialEpochs.mat', squeeze_me=True)
    dat1.keys()
    x = dat1['ep_up_mean']  #All coherent and incoherent stims 
    y = dat1['ep_down_mean']  #All coherent and incoherent stims
    sfreq=dat1['sfreq']
    t_resp = np.arange(0,y[0].size/sfreq,1/sfreq)
    
    # y_all += [y,]
    
    x_avg = x.mean(axis=0)
    plt.plot(t_resp, x_avg)
    plt.show()
    #Extracting incoherent and coherent components of response
    t_respincoh1 = t_resp[(t_resp>=0) & (t_resp<1)]
    # y1=y[0][(t_resp>=0) & (t_resp<=1)] #Single trial
    t_respcoh1 = t_resp[(t_resp>=1) & (t_resp<2)]
    # y2=y[0][(t_resp>=1) & (t_resp<=2)] #Single trial 
    t_respincoh2 = t_resp[(t_resp>=2) & (t_resp<3)]
    # y3=y[0][(t_resp>=2) & (t_resp<=3)] #Single trial
    t_respcoh2 = t_resp[(t_resp>=3) & (t_resp<4)]
    # y4=y[0][(t_resp>=3) & (t_resp<=4)] #Single trial 
    t_respincoh3 = t_resp[(t_resp>=4) & (t_resp<5)]
    # y5=y[0][(t_resp>=4) & (t_resp<=5)] #Single trial
 
    r_coh1=np.zeros((len(y),len(t_respcoh1)))
    r_coh2=np.zeros((len(y),len(t_respcoh2)))
    r_incoh1=np.zeros((len(y),len(t_respincoh1)))
    r_incoh2=np.zeros((len(y),len(t_respincoh2)))
    r_incoh3=np.zeros((len(y),len(t_respincoh3)))

    for c in range(len(y)):
        r_coh1[c,]=y[c][(t_resp>=1) & (t_resp<2)]
        r_coh2[c,]=y[c][(t_resp>=3) & (t_resp<4)]
        r_incoh1[c,]=y[c][(t_resp>=0) & (t_resp<1)]
        r_incoh2[c,]=y[c][(t_resp>=2) & (t_resp<3)]
        r_incoh3[c,]=y[c][(t_resp>=4) & (t_resp<5)]  
        

#Averaging the incoherent and coherent parts only for plotting later 
    r_coh1_avg = r_coh1.mean(axis=0)
    r_coh2_avg = r_coh2.mean(axis=0)
    r_incoh1_avg = r_incoh1.mean(axis=0)
    r_incoh2_avg = r_incoh2.mean(axis=0)
    r_incoh3_avg = r_incoh3.mean(axis=0)
    # a = t_respincoh1[t_respincoh1>0.05]
    # b = r_incoh1_avg[t_respincoh1>0.05]

#Computing noise floor 
    a= (-1)*r_coh1[::2]
    b=(r_coh1[1::2])
    
    n_up = (((ep_up_mean[1::2]) + (-1)*ep_up_mean[::2])).mean(axis=0)
    
    n_up = (np.pad((r_coh1[1::2]), [(0,1), (0,0)], mode='constant') + (-1)*r_coh1[::2]).mean(axis=0)
    # n_coh1 = (r_coh1[1::2]).mean(axis=0) + ((-1)*r_coh1[::2]).mean(axis=0)
    # n_coh1 = a + b
    # n_coh1 = np.flip(r_coh1[::2],axis=0).mean(axis=0)
    n_coh2 = (np.pad((r_coh2[1::2]), [(0,1), (0,0)], mode='constant') + (-1)*r_coh2[::2]).mean(axis=0)
    n_incoh1 = (np.pad((r_incoh1[1::2]), [(0,1), (0,0)], mode='constant') + (-1)*r_incoh1[::2]).mean(axis=0)
    n_incoh2 = (np.pad((r_incoh2[1::2]), [(0,1), (0,0)], mode='constant') + (-1)*r_incoh2[::2]).mean(axis=0)
    n_incoh3 = (np.pad((r_incoh3[1::2]), [(0,1), (0,0)], mode='constant') + (-1)*r_incoh3[::2]).mean(axis=0)
    
    
    n_coh2 = (r_coh2[1::2]).mean(axis=0) + ((-1)*r_coh2[::2]).mean(axis=0)
    n_incoh1 = (r_incoh1[1::2]).mean(axis=0) + ((-1)*r_incoh1[::2]).mean(axis=0)
    n_incoh2 = (r_incoh2[1::2]).mean(axis=0) + ((-1)*r_incoh2[::2]).mean(axis=0)
    n_incoh3 = (r_incoh3[1::2]).mean(axis=0) + ((-1)*r_incoh3[::2]).mean(axis=0)
    
    # n_floor= n_coh1 + n_coh2 + n_incoh1 + n_incoh2 + n_incoh3
    
    plt.plot(t_respcoh1,r_coh1_avg, label='resp', color='black')
    plt.plot(t_respcoh1, n_coh1, color ='grey')
    # plt.plot(t_respcoh1, n_coh1_1, color ='blue')
    # plt.plot(t_respcoh1, n_coh1_1)
    plt.plot(t_respcoh2, r_coh2_avg, color='black')
    plt.plot(t_respcoh2, n_coh2, color ='grey' )
    plt.plot(t_respincoh1,r_incoh1_avg, color='black')
    plt.plot(t_respincoh1,n_incoh1, color ='grey')
    plt.plot(t_respincoh2,r_incoh2_avg, color='black')
    plt.plot(t_respincoh2,n_incoh2, color ='grey')
    plt.plot(t_respincoh3,r_incoh3_avg, color='black')
    plt.plot(t_respincoh3,n_incoh3, label='noise', color ='grey')
    plt.legend()
    plt.show()
    
    n_up = x_avg - n_up
    plt.plot(t,n_up, label ='noise')
    plt.plot(t,x_avg, label ='EFR')
    plt.title('Pitch Up')
    plt.legend()
    plt.show()


    
    mat_ids_resp = dict(r_coh1_avg=r_coh1_avg, r_coh2_avg=r_coh2_avg, r_incoh1_avg=r_incoh1_avg, r_incoh2_avg=r_incoh2_avg, r_incoh3_avg=r_incoh3_avg,
                        n_coh1=n_coh1, n_coh2=n_coh2, n_incoh1=n_incoh1,n_incoh2=n_incoh2,n_incoh3=n_incoh3, t_respincoh1=t_respincoh1, t_respcoh1= t_respcoh1,
                        t_respincoh2=t_respincoh2, t_respcoh2=t_respcoh2,t_respincoh3=t_respincoh3) 
    savemat(save_loc_mat + subj + '_respnoise_NoRej.mat' , mat_ids_resp)
    ###FFT
    # x3f = np.fft.fft(ep_mean_all,axis=0)
    # N3 = x3f.size
    # f3 = np.fft.fftfreq(N3, d=1/sfreq) # get frequency vector
    # #Just look at postivie frequencies since fft is symmetric for real signals
    # x3f = x3f[f3>=0]
    # f3 = f3[f3>=0]
    # plt.plot(f3,abs(x3f), label='EP-Coherence 1')
    # plt.xlabel('Frequency (Hz)',fontsize=14)
    # plt.ylabel('Magnitude',fontsize=14)
    # #ax[0].set_xlim([0,1000])
    # plt.title('Uncorrelated -- FFT',fontweight='bold',fontsize=14)
    # plt.show()
  
#%%% Cross correlation of stimulus and responses
###Individual coh and incoh
corr1=np.zeros((len(r_incoh1), (2*tcoh1.size-1)))  #It's technically 2*sfreq-1 (the end corr is always odd)                                               
lags1= np.zeros((len(r_incoh1), (2*tcoh1.size-1)))
corr2=np.zeros((len(r_coh1),2*tcoh1.size-1))
corr3=np.zeros((len(r_incoh2),2*tcoh1.size-1))
corr4=np.zeros((len(r_coh2), 2*tcoh1.size-1))
corr5=np.zeros((len(r_incoh3), 2*tcoh1.size-2))
lags5= np.zeros((corr5.shape))

for c in range(len(y)):
    corr1[c,] = signal.correlate(r_incoh1[c], amp1[c], mode='full')
    corr2[c,] = signal.correlate(r_coh1[c], amp2[c])
    corr3[c,] = signal.correlate(r_incoh2[c], amp3[c])
    corr4[c,] = signal.correlate(r_coh2[c], amp4[c])
    corr5[c,] = signal.correlate(r_incoh3[c], amp5[c])

#Calculating lags (Will remain same for each trial, hence considering only the first trial here)

lags1= signal.correlation_lags(r_coh1[0].size,amp1[0].size, mode='full')
lags5= signal.correlation_lags(r_incoh3[0].size, amp5[0].size)

# lag1 = lags1[np.argmax(corr1)]

# corr1 /=np.max(corr1)

plt.plot(lags1,corr1[0], label ='Incoherent 1')
plt.plot(lags1,corr2[0],  label ='Coherent 1')
plt.plot(lags1,corr3[0],  label ='Incoherent 2')
plt.plot(lags1,corr4[0],  label ='Coherent 2')
plt.plot(lags5,corr5[0],  label ='Incoherent 3')
plt.xlabel('Lag Time (s)',fontsize=14)
plt.ylabel('Cross-correlation (Avg)',fontsize=14)
# plt.xlim(-7,7)
plt.title ('Method = Full')
plt.legend()
plt.show()

# corr_incoh_avg = corr_incoh.mean(axis=0)
# corr_coh_avg = corr_coh.mean(axis=0)

lags1 = lags1/sfreq
lags5 =lags5/sfreq

corr1_avg = corr1.mean(axis=0)/np.max(corr1)
corr2_avg = corr2.mean(axis=0)/np.max(corr2)
corr3_avg = corr3.mean(axis=0)/np.max(corr3)
corr4_avg = corr4.mean(axis=0)/np.max(corr4)
corr5_avg = corr5.mean(axis=0)/np.max(corr5)

corr1_med = np.median(corr1,axis=0)
corr2_med = np.median(corr2,axis=0)
corr3_med = np.median(corr3,axis=0)
corr4_med = np.median(corr4,axis=0)
corr5_med = np.median(corr5,axis=0)

# corr1_n = corr1_avg/np.max(corr1)
# corr2_n = corr2_avg/np.max(corr2)

corr_incoh = corr1_avg + corr3_avg
corr_coh = corr2_avg +corr4_avg

corr_incoh_med=corr1_med+corr3_med
corr_coh_med=corr2_med+corr4_med

###Saving matfiles

mat_ids_corr = dict(corr1_avg=corr1_avg, corr2_avg=corr2_avg, corr3_avg=corr3_avg, corr4_avg=corr4_avg,
                    corr5_avg=corr5_avg, sfreq=sfreq, lags1=lags1, lags5=lags5)
io.savemat(save_loc_mat + subj + '_corr_stats.mat', mat_ids_corr) 

mat_ids_med = dict(corr1_med=corr1_med, corr2_med=corr2_med, corr3_med=corr3_med, corr4_med=corr4_med,
                    corr5_med=corr5_med, sfreq=sfreq, lags1=lags1, lags5=lags5, corr_incoh_med=corr_incoh_med, corr_coh_med=corr_coh_med)
io.savemat(save_loc_mat + subj + '_corr_median.mat', mat_ids_med) 

####Plot all cross-correlations with lag 
plt.plot(lags1,corr1_avg, label ='Incoherent')
plt.plot(lags1,corr2_avg,  label ='Coherent')
plt.plot(lags1,corr3_avg, label ='Incoherent')
plt.plot(lags1,corr4_avg,  label ='Coherent')
plt.plot(lags5,corr5_avg, label ='Incoherent')
plt.plot(lags1, corr_incoh,linestyle='dashed', color='black',linewidth=5, label = 'Mean Incoherent')
plt.plot(lags1, corr_coh,linestyle='dashed', color='grey',linewidth=5, label = 'Mean Coherent')
plt.xlabel('Lag Time (s)',fontsize=14)
plt.ylabel('Cross-correlation (Avg)',fontsize=14)
# plt.xlim(-7,7)
plt.title ( subj + '- Cross Correlations -- Method = Full')
plt.tight_layout()
plt.legend()
plt.show()

plt.savefig(save_loc_fig + subj + '_CrossCorr_All_withmean', dpi=500)


plt.plot(lags1,corr1_med, label ='Incoherent')
plt.plot(lags1,corr2_med,  label ='Coherent')
plt.plot(lags1,corr3_med, label ='Incoherent')
plt.plot(lags1,corr4_med,  label ='Coherent')
plt.plot(lags5,corr5_med, label ='Incoherent')
plt.plot(lags1, corr_incoh_med,linestyle='dashed', color='black',linewidth=5, label = 'Mean Incoherent')
plt.plot(lags1, corr_coh_med,linestyle='dashed', color='grey',linewidth=5, label = 'Mean Coherent')
plt.xlabel('Lag Time (s)',fontsize=14)
plt.ylabel('Cross-correlation (Median)',fontsize=14)
# plt.xlim(-7,7)
plt.title ( subj + '- Cross Correlations -- Method = Full')
plt.tight_layout()
plt.legend()
plt.show()

plt.savefig(save_loc_fig + subj + '_CrossCorr_All_withmedian', dpi=500)

# ##Plot incoherent and coherent stim, response, cross-correlations individually 
# fig, (ax_incoh, ax_coh, ax_resp, ax_corr) = plt.subplots(4, 1, figsize=(6,5))
# ax_incoh.plot(tincoh1, incoh1[0], label ='Incoherent - Original Signal')
# ax_incoh.set_title('Original Incoherent signal - Example')
# ax_incoh.set_xlabel('Time(s)')
# ax_incoh.plot(tincoh1, amp1[0], label = 'Hilbert transform')
# ax_coh.plot(tcoh1, coh1[0], label ='Coherent - Original Signal')
# ax_coh.set_title('Original Coherent signal - Example')
# ax_coh.set_xlabel('Time(s)')
# ax_coh.plot(tcoh1, amp2[0], label = 'Hilbert transform')
# ax_resp.plot(t_respincoh1, r_incoh1[0], label ='Incoherent')
# ax_resp.plot(t_respincoh1,r_coh1[0], label ='Coherent')
# ax_resp.set_title('EFR')
# ax_resp.set_xlabel('Time (s)')
# ax_corr.plot(lags1, corr1[0], label = 'Incoherence Corr')
# ax_corr.plot(lags1, corr2[0], label = 'Coherence Corr')
# ax_corr.set_title('Cross-correlated signal')
# ax_corr.set_xlabel('Lag (s)')
# # ax_orig.margins(0, 0.1)
# # ax_noise.margins(0, 0.1)
# # ax_corr.margins(0, 0.1)
# fig.tight_layout()
# plt.legend()
# plt.show()

# plt.savefig(save_loc_fig + subj+  '_All_EFR_CrossCorr_1', dpi=500)

# ###Plotting EFR and cross correlations
# lags1=lags1/sfreq
# lags2=lags2/sfreq
# lags3=lags3/sfreq
# lags4=lags4/sfreq
# lags5=lags5/sfreq

fig,ax = plt.subplots(2,1,figsize=(7,5), constrained_layout=False)
ax[0].plot(t_respincoh1,r_incoh1_avg, label='resp')
ax[0].plot(t_respincoh1,n_incoh1, color ='grey', label='noise')
ax[0].plot(t_respcoh1,r_coh1_avg)
ax[0].plot(t_respcoh1, n_coh1, color ='grey')
ax[0].plot(t_respincoh2,r_incoh2_avg)
ax[0].plot(t_respincoh2,n_incoh2, color ='grey')
ax[0].plot(t_respcoh2,r_coh2_avg)
ax[0].plot(t_respcoh2, n_coh2, color ='grey' )
ax[0].plot(t_respincoh3,r_incoh3_avg)
ax[0].plot(t_respincoh3,n_incoh3, label='noise', color ='grey')
ax[0].set_title('EFR?')
ax[0].set_xlabel('Time(s)')
ax[0].set_ylabel('Amplitude(' + u"\u03bcA" + ')')
ax[1].plot(lags1,corr1_avg, label ='Incoherent 1')
ax[1].plot(lags1,corr2_avg,  label ='Coherent 1')
ax[1].plot(lags1,corr3_avg,  label ='Incoherent 2')
ax[1].plot(lags1,corr4_avg,  label ='Coherent 2')
ax[1].plot(lags5,corr5_avg,  label ='Incoh3')
ax[1].plot(lags1, corr_incoh,linestyle='dashed', color='black',linewidth=5, label = 'Mean Incoherent')
ax[1].plot(lags1, corr_coh,linestyle='dashed', color='grey',linewidth=5, label = 'Mean Coherent')
ax[1].set_title('Cross-Correlations')
ax[1].set_xlabel('Lag Time (s)')
ax[1].set_ylabel('Cross-correlation (Avg)')
fig.subplots_adjust(bottom=0.75) 
# fig.subplots_adjust(right=0.5)
fig.legend(loc='lower center', ncol=7, prop={'size': 6})
fig.tight_layout()
fig.suptitle(subj + '- EFR and Cross-Correlations', fontweight='bold',fontsize=14)
plt.show()

plt.savefig(save_loc_fig + subj +  '_EFR_CrossCorr_All', dpi=500)


fig,ax = plt.subplots(2,1,figsize=(7,5), constrained_layout=False)
ax[0].plot(t_respincoh1,r_incoh1_avg, label='resp')
ax[0].plot(t_respincoh1,n_incoh1, color ='grey', label='noise')
ax[0].plot(t_respcoh1,r_coh1_avg)
ax[0].plot(t_respcoh1, n_coh1, color ='grey')
ax[0].plot(t_respincoh2,r_incoh2_avg)
ax[0].plot(t_respincoh2,n_incoh2, color ='grey')
ax[0].plot(t_respcoh2,r_coh2_avg)
ax[0].plot(t_respcoh2, n_coh2, color ='grey' )
ax[0].plot(t_respincoh3,r_incoh3_avg)
ax[0].plot(t_respincoh3,n_incoh3, label='noise', color ='grey')
ax[0].set_title('EFR?')
ax[0].set_xlabel('Time(s)')
ax[0].set_ylabel('Amplitude(' + u"\u03bcA" + ')')
ax[1].plot(lags1,corr1_med, label ='Incoherent 1')
ax[1].plot(lags1,corr2_med,  label ='Coherent 1')
ax[1].plot(lags1,corr3_med,  label ='Incoherent 2')
ax[1].plot(lags1,corr4_med,  label ='Coherent 2')
ax[1].plot(lags5,corr5_med,  label ='Incoh3')
ax[1].plot(lags1, corr_incoh_med,linestyle='dashed', color='black',linewidth=5, label = 'Mean Incoherent')
ax[1].plot(lags1, corr_coh_med,linestyle='dashed', color='grey',linewidth=5, label = 'Mean Coherent')
ax[1].set_title('Cross-Correlations')
ax[1].set_xlabel('Lag Time (s)')
ax[1].set_ylabel('Cross-correlation (Median)')
fig.subplots_adjust(bottom=0.75) 
# fig.subplots_adjust(right=0.5)
fig.legend(loc='lower center', ncol=7, prop={'size': 6})
fig.tight_layout()
fig.suptitle(subj + '- EFR and Cross-Correlations', fontweight='bold',fontsize=14)
plt.show()


###Full 5 second - Not individual coherent and incoherent
# corr=np.zeros((len(y), 85186))
# corr_norm=np.zeros((len(y), 85186))
# # lags= np.zeros((corr1.shape))

# for c in range(len(y)):
#     corr[c,] = signal.correlate(amp[c], y[c])
#     corr_norm[c,] = corr[c]/np.max(corr)
    
# lags= signal.correlation_lags(y[0].size,amp[0].size)
# lags=lags/sfreq   

# corr_avg=corr.mean(axis=0)
# corr_norm_avg = corr_norm.mean(axis=0)

# lag = lags[np.argmax(corr_avg)]

# plt.plot(lags,corr_norm_avg)
# plt.xlabel('Lag Time (s)',fontsize=14)
# plt.ylabel('Cross-correlation (Avg)',fontsize=14)
# plt.xlim(-7,7)
# plt.title ('Method = Full')
# plt.show()

# plt.plot(lags, corr_norm[100])
# plt.xlabel('Lags',fontsize=14)
# plt.ylabel('Cross-correlation (Avg)',fontsize=14)
# plt.xlim(-7,7)
# plt.show()

# fig,ax = plt.subplots(3,1,figsize=(8,12))
# ax[0].plot(amp1[0])
# ax[0].set_title('Original signal',fontweight='bold',fontsize=14)
# ax[1].plot(y[0])
# ax[1].set_title('EFR?',fontweight='bold',fontsize=14)
# ax[2].plot(corr1)
# ax[2].set_title('Correlated',fontweight='bold',fontsize=14)
# fig.suptitle('Hilbert transform-Averaged across trials', fontweight='bold',fontsize=14)
# fig.show()  
    
# fig, (ax_orig, ax_noise, ax_corr) = plt.subplots(3, 1, figsize=(6,5))
# ax_orig.plot(t, x[0], label ='Original Signal')
# ax_orig.set_title('Original signal - Example')
# ax_orig.set_xlabel('Time(s)')
# ax_orig.plot(t, amp[0], label = 'Hilbert transform')
# ax_noise.plot(t_resp, y_avg)
# ax_noise.set_title('EFR')
# ax_noise.set_xlabel('Time (s)')
# ax_corr.plot(lags, corr_avg)
# ax_corr.set_title('Cross-correlated signal')
# ax_corr.set_xlabel('Lag (s)')
# # ax_orig.margins(0, 0.1)
# # ax_noise.margins(0, 0.1)
# # ax_corr.margins(0, 0.1)
# fig.tight_layout()
# fig.suptitle(subj + ' - Awake', fontweight='bold',fontsize=14)
# plt.legend()
# plt.show()

# plt.savefig(save_loc_fig + 'EFR_CrossCorr', dpi=500)

#%% Pooling across subjects 
r_coh1_avg = []
r_coh2_avg = []
r_incoh1_avg = []
r_incoh2_avg = []
r_incoh3_avg = []
n_coh1 = []
n_coh2 = []
n_incoh1 = []
n_incoh2 = []
n_incoh3 = []

subjlist = ['Q414_2', 'Q419']  # Load subject folder
for subj in subjlist:    
    dat2 = io.loadmat(save_loc_mat + subj + '_respnoise_NoRej.mat', squeeze_me=True)
    dat2.keys()
    a = dat2['r_coh1_avg']
    b = dat2['r_coh2_avg']
    c = dat2['r_incoh1_avg']
    d = dat2['r_incoh2_avg']
    e = dat2['r_incoh3_avg']
    f = dat2['n_coh1']
    g = dat2['n_coh2']
    h = dat2['n_incoh1']
    i = dat2['n_incoh2']
    j = dat2['n_incoh3']
    t1 = dat2['t_respincoh1']
    t2 = dat2['t_respcoh1']
    t3 = dat2['t_respincoh2']
    t4 =dat2['t_respcoh2']
    t5 = dat2['t_respincoh3']
   
    r_coh1_avg += [a,]
    r_coh2_avg += [b,]
    r_incoh1_avg += [c,]
    r_incoh2_avg += [d,]
    r_incoh3_avg += [e,]
    n_coh1 += [f,]
    n_coh2 += [g,]
    n_incoh1 += [h,]
    n_incoh2 += [i,]
    n_incoh3 += [j,]
    
# x = np.array(r_coh1_avg)
    
r_coh1_all = (np.array(r_coh1_avg)).mean(axis=0)
r_coh1_sem = sem(r_coh1_avg)
r_coh2_all = (np.array(r_coh2_avg)).mean(axis=0)
r_coh2_sem = sem(r_coh2_avg)
r_incoh1_all = (np.array(r_incoh1_avg)).mean(axis=0)
r_incoh1_sem = sem(r_incoh1_avg)
r_incoh2_all = (np.array(r_incoh2_avg)).mean(axis=0)
r_incoh2_sem = sem(r_incoh2_avg)
r_incoh3_all = (np.array(r_incoh3_avg)).mean(axis=0)
r_incoh3_sem = sem(r_incoh3_avg)
n_coh1_all = (np.array(n_coh1)).mean(axis=0)
n_coh1_sem = sem(n_coh1)
n_coh2_all = (np.array(n_coh2)).mean(axis=0)
n_coh2_sem = sem(n_coh2)
n_incoh1_all = (np.array(n_incoh1)).mean(axis=0)
n_incoh1_sem = sem(n_incoh1)
n_incoh2_all = (np.array(n_incoh2)).mean(axis=0)
n_incoh2_sem = sem(n_incoh2)
n_incoh3_all = (np.array(n_incoh3)).mean(axis=0)
n_incoh3_sem = sem(n_incoh3)

plt.errorbar(t1,r_incoh1_all,yerr=r_incoh1_sem, color='black', linewidth=2, ecolor='darkgray', label='EFR')
plt.errorbar(t2,r_coh1_all,yerr=r_coh1_sem, color='black', linewidth=2, ecolor='darkgray')
plt.errorbar(t3,r_incoh2_all,yerr=r_incoh2_sem, color='black', linewidth=2, ecolor='darkgray')
plt.errorbar(t4,r_coh2_all,yerr=r_coh2_sem, color='black', linewidth=2, ecolor='darkgray')
plt.errorbar(t5,r_incoh3_all,yerr=r_incoh3_sem, color='black', linewidth=2, ecolor='darkgray')
plt.errorbar(t1,n_incoh1_all,yerr=n_incoh1_sem, color='grey', linewidth=2, ecolor='lightgrey', label='Noise floor')
plt.errorbar(t2,n_coh1_all,yerr=n_coh1_sem, color='grey', linewidth=2, ecolor='lightgrey')
plt.errorbar(t3,n_incoh2_all,yerr=n_incoh2_sem, color='grey', linewidth=2, ecolor='lightgrey')
plt.errorbar(t4,n_coh2_all,yerr=n_coh2_sem, color='grey', linewidth=2, ecolor='lightgrey')
plt.errorbar(t5,n_incoh3_all,yerr=n_incoh3_sem, color='grey', linewidth=2, ecolor='lightgrey')
plt.legend()
plt.show()


corr1_avg_all =[]
corr2_avg_all =[]
corr3_avg_all =[]
corr4_avg_all = []
corr5_avg_all =[]

for subj in subjlist:
    dat3 = io.loadmat(save_loc_mat + subj + '_corr_stats.mat', squeeze_me=True)
    dat3.keys()
    corr1_avg = dat3['corr1_avg']
    corr2_avg=dat3['corr2_avg']
    corr3_avg=dat3['corr3_avg']
    corr4_avg=dat3['corr4_avg']
    corr5_avg=dat3['corr5_avg']
    sfreq=dat3['sfreq']
    lags1=dat3['lags1']
    lags5=dat3['lags5']
    
    corr1_avg_all +=[corr1_avg,]
    corr2_avg_all +=[corr2_avg,]
    corr3_avg_all +=[corr3_avg,]
    corr4_avg_all += [corr4_avg,]
    corr5_avg_all +=[corr5_avg,]
    
corr1_avg_alls = (np.array(corr1_avg_all)).mean(axis=0)
corr2_avg_alls = (np.array(corr2_avg_all)).mean(axis=0)
corr3_avg_alls = (np.array(corr3_avg_all)).mean(axis=0)
corr4_avg_alls = (np.array(corr4_avg_all)).mean(axis=0)
# corr5_avg_all = (np.array(corr5_avg_all)).mean(axis=0)

corr_incoh =corr1_avg_alls + corr3_avg_alls #+corr5_avg_all
corr_incoh_sem = sem(corr1_avg_all) + sem(corr3_avg_all)
corr_coh = corr2_avg_alls + corr4_avg_alls
corr_coh_sem = sem(corr2_avg_all + corr4_avg_all)

# plt.plot(lags1,corr1_avg, label ='Incoherent')
# plt.plot(lags1,corr2_avg,  label ='Coherent')
# plt.plot(lags1,corr3_avg, label ='Incoherent')
# plt.plot(lags1,corr4_avg,  label ='Coherent')
# plt.plot(lags5,corr5_avg, label ='Incoherent')
plt.plot(lags1, corr_incoh,linestyle='dashed', color='black',linewidth=5, label = 'Mean Incoherent')
plt.plot(lags1, corr_coh,linestyle='dashed', color='grey',linewidth=5, label = 'Mean Coherent')
# plt.plot(lags1,corr1_avg, label ='Incoherent 1')
# plt.plot(lags2,corr2_avg,  label ='Coherent 1')
# plt.plot(lags3,corr3_avg,  label ='Incoherent 2')
# plt.plot(lags4,corr4_avg,  label ='Coherent 2')
# plt.plot(lags5,corr5_avg,  label ='Incoherent 3')
plt.xlabel('Lag Time (s)',fontsize=14)
plt.ylabel('Cross-correlation (Avg)',fontsize=14)
# plt.xlim(-7,7)
plt.title ( subj + '- Cross Correlations -- Method = Full')
plt.tight_layout()
plt.legend()
plt.show()

fig,ax = plt.subplots(2,1,figsize=(7,5), constrained_layout=False)
ax[0].errorbar(t1,r_incoh1_all,yerr=r_incoh1_sem, color='black', linewidth=2, ecolor='darkgray', label='EFR')
ax[0].errorbar(t2,r_coh1_all,yerr=r_coh1_sem, color='black', linewidth=2, ecolor='darkgray')
ax[0].errorbar(t3,r_incoh2_all,yerr=r_incoh2_sem, color='black', linewidth=2, ecolor='darkgray')
ax[0].errorbar(t4,r_coh2_all,yerr=r_coh2_sem, color='black', linewidth=2, ecolor='darkgray')
ax[0].errorbar(t5,r_incoh3_all,yerr=r_incoh3_sem, color='black', linewidth=2, ecolor='darkgray')
ax[0].errorbar(t1,n_incoh1_all,yerr=n_incoh1_sem, color='grey', linewidth=2, ecolor='lightgrey', label='Noise floor')
ax[0].errorbar(t2,n_coh1_all,yerr=n_coh1_sem, color='grey', linewidth=2, ecolor='lightgrey')
ax[0].errorbar(t3,n_incoh2_all,yerr=n_incoh2_sem, color='grey', linewidth=2, ecolor='lightgrey')
ax[0].errorbar(t4,n_coh2_all,yerr=n_coh2_sem, color='grey', linewidth=2, ecolor='lightgrey')
ax[0].errorbar(t5,n_incoh3_all,yerr=n_incoh3_sem, color='grey', linewidth=2, ecolor='lightgrey')
ax[0].set_title('EFR?')
ax[0].set_xlabel('Time(s)')
ax[0].set_ylabel('Amplitude(' + u"\u03bcA" + ')')
ax[1].errorbar(lags1,corr_incoh,yerr=corr_incoh_sem, color='teal', linewidth=2, ecolor='lightcyan', label='Incoherent')
ax[1].errorbar(lags1,corr_coh,yerr=corr_coh_sem, color='maroon', linewidth=2, ecolor='lightcoral', label='Coherent')
ax[1].set_title('Cross-Correlations')
ax[1].set_xlabel('Lag Time (s)')
ax[1].set_ylabel('Cross-correlation (Avg)')
ax[0].legend(loc='upper right')
ax[1].legend(loc='lower right')
fig.tight_layout()
plt.subplots_adjust(top=0.85)
fig.suptitle('Awake EFR and Cross-Correlations (N=' + str(len(subjlist)) + ')', fontweight='bold',fontsize=14)
plt.show()

plt.savefig(save_loc_fig + 'All_EFR_CrossCorr.png', dpi=500)


corr1_med_all =[]
corr2_med_all =[]
corr3_med_all =[]
corr4_med_all = []
corr5_med_all =[]

for subj in subjlist:
    dat3 = io.loadmat(save_loc_mat + subj + '_corr_median.mat', squeeze_me=True)
    dat3.keys()
    corr1_med = dat3['corr1_med']
    corr2_med=dat3['corr2_med']
    corr3_med=dat3['corr3_med']
    corr4_med=dat3['corr4_med']
    corr5_med=dat3['corr5_med']
    sfreq=dat3['sfreq']
    lags1=dat3['lags1']
    lags5=dat3['lags5']
    
    corr1_med_all +=[corr1_med,]
    corr2_med_all +=[corr2_med,]
    corr3_med_all +=[corr3_med,]
    corr4_med_all += [corr4_med,]
    corr5_med_all +=[corr5_med,]
    
corr1_med_alls = (np.array(corr1_med_all)).mean(axis=0)
corr2_med_alls = (np.array(corr2_med_all)).mean(axis=0)
corr3_med_alls = (np.array(corr3_med_all)).mean(axis=0)
corr4_med_alls = (np.array(corr4_med_all)).mean(axis=0)
# corr5_avg_all = (np.array(corr5_avg_all)).mean(axis=0)

corr_incoh_med =corr1_med_alls + corr3_med_alls #+corr5_med_all
corr_incoh_sem_med = sem(corr1_med_all) + sem(corr3_med_all)
corr_coh_med = corr2_med_alls + corr4_med_alls
corr_coh_sem_med = sem(corr2_med_all + corr4_med_all)

fig,ax = plt.subplots(2,1,figsize=(7,5), constrained_layout=False)
ax[0].errorbar(t1,r_incoh1_all,yerr=r_incoh1_sem, color='black', linewidth=2, ecolor='darkgray', label='EFR')
ax[0].errorbar(t2,r_coh1_all,yerr=r_coh1_sem, color='black', linewidth=2, ecolor='darkgray')
ax[0].errorbar(t3,r_incoh2_all,yerr=r_incoh2_sem, color='black', linewidth=2, ecolor='darkgray')
ax[0].errorbar(t4,r_coh2_all,yerr=r_coh2_sem, color='black', linewidth=2, ecolor='darkgray')
ax[0].errorbar(t5,r_incoh3_all,yerr=r_incoh3_sem, color='black', linewidth=2, ecolor='darkgray')
ax[0].errorbar(t1,n_incoh1_all,yerr=n_incoh1_sem, color='grey', linewidth=2, ecolor='lightgrey', label='Noise floor')
ax[0].errorbar(t2,n_coh1_all,yerr=n_coh1_sem, color='grey', linewidth=2, ecolor='lightgrey')
ax[0].errorbar(t3,n_incoh2_all,yerr=n_incoh2_sem, color='grey', linewidth=2, ecolor='lightgrey')
ax[0].errorbar(t4,n_coh2_all,yerr=n_coh2_sem, color='grey', linewidth=2, ecolor='lightgrey')
ax[0].errorbar(t5,n_incoh3_all,yerr=n_incoh3_sem, color='grey', linewidth=2, ecolor='lightgrey')
ax[0].set_title('EFR?')
ax[0].set_xlabel('Time(s)')
ax[0].set_ylabel('Amplitude(' + u"\u03bcA" + ')')
ax[1].errorbar(lags1,corr_incoh_med,yerr=corr_incoh_sem_med, color='teal', linewidth=2, ecolor='lightcyan', label='Incoherent')
ax[1].errorbar(lags1,corr_coh_med,yerr=corr_coh_sem_med, color='maroon', linewidth=2, ecolor='lightcoral', label='Coherent')
ax[1].set_title('Cross-Correlations')
ax[1].set_xlabel('Lag Time (s)')
ax[1].set_ylabel('Cross-correlation (Median)')
ax[0].legend(loc='lower right')
ax[1].legend(loc='lower right')
fig.tight_layout()
plt.subplots_adjust(top=0.85)
fig.suptitle('Awake EFR and Cross-Correlations (N=' + str(len(subjlist)) + ')', fontweight='bold',fontsize=14)
plt.show()

plt.savefig(save_loc_fig + 'All_EFR_CrossCorr_Median.png', dpi=500)
