# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:23:11 2023

@author: vmysorea
"""

###Saving steady state amp, peak amp and latencies of P1, P2 for all conditions 
import sys
sys.path.append('C:/Users/vmysorea/Documents/mne-python/')
import warnings
from matplotlib import pyplot as plt
from scipy.io import savemat
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
save_fig = 'C:/Users/vmysorea/Desktop/PhD/GreenLightMeeting/Figures/'
save_mat_loc = 'D:/PhD/Data/Binding_matfiles/0.1-40Hz/'

subjlist = ['S273', 'S069', 'S072', 'S078', 'S088', 
            'S105', 'S207', 'S259', 'S260', 'S268', 
            'S269', 'S270', 'S271', 'S272', 'S274', 
            'S277', 'S279', 
            'S280', 'S281', 'S282', 'S284', 'S285', 
            'S288', 'S290', 'S291', 'S303', 'S305', 
            'S308', 'S309', 'S310', 'S312', 'S337', 
            'S339', 'S340', 'S341', 'S342', 'S344', 
            'S345', 'S347', 'S352', 'S355', 'S358']

#Uncommon == 'S105', 'S272', 'S309', 'S337', 'S345','S355', 'S358'

#Removed --  'S273'-- Looks like equal amount of positive and negative components for GFP ; 
#            'S104' -- NAN
#           'S358' -- Last incoherent period weird 
#'S345' also weird? -- Extremely high activity at last incoherent period -- 400 ms 

#%% Loading all subjects' data 

# 0 - 12 Onset (Upto 1.1 s) -- 0
# 1 - 20 Onset (Upto 1.1 s) -- 1
# 10 - 12 Incoherent to Coherent -- 2
# 11 - 12 Coherent to Incoherent -- 3
# 12 - 20 Incoherent to Coherent -- 4
# 13 - 20 Coherent to Incoherent -- 5
# 14 - 12 Full 5 seconds -- 6
# 15 - 20 Full 5 seconds -- 7 

for subj in range(len(subjlist)):
    sub = subjlist [subj]
    dat = io.loadmat(save_mat_loc + sub + '_1-40Hz_Evoked_AllChan.mat', squeeze_me=True)
    dat.keys()
    t = dat['t']
    t_full = dat ['t_full']
    
gfps0 = np.zeros((len(subjlist),t.size))
gfps1 = np.zeros((len(subjlist),t.size))
gfps2 = np.zeros((len(subjlist),t.size))
gfps3 = np.zeros((len(subjlist),t.size))
gfps4 = np.zeros((len(subjlist),t.size))
gfps5 = np.zeros((len(subjlist),t.size))
gfps6 = np.zeros((len(subjlist),t_full.size))
gfps7 = np.zeros((len(subjlist),t_full.size))

evkds0 = np.zeros((len(subjlist),t.size))
evkds1 = np.zeros((len(subjlist),t.size))
evkds2 = np.zeros((len(subjlist),t.size))
evkds3 = np.zeros((len(subjlist),t.size))
evkds4 = np.zeros((len(subjlist),t.size))
evkds5 = np.zeros((len(subjlist),t.size))
evkds6 = np.zeros((len(subjlist),t_full.size))
evkds7 = np.zeros((len(subjlist),t_full.size))

picks = [4, 25, 30, 31]
    
for subj in range(len(subjlist)):
    sub = subjlist [subj]
    dat = io.loadmat(save_mat_loc + sub + '_1-40Hz_Evoked_AllChan.mat', squeeze_me=True)
    dat.keys()    
    evkd0 = dat['evkd0'][picks]
    evkd1 = dat['evkd1'][picks]
    evkd2 = dat['evkd2'][picks]
    evkd3 = dat['evkd3'][picks]
    evkd4 = dat['evkd4'][picks]
    evkd5 = dat['evkd5'][picks]
    evkd6 = dat['evkd6'][picks]
    evkd7 = dat['evkd7'][picks]
    gfps0[subj,:] = evkd0[0:31].std(axis=0)     #Taking the SD of the first 32 channels, excluding the EXGs
    gfps1[subj,:] = evkd1[0:31].std(axis=0) 
    gfps2[subj,:] = evkd2[0:31].std(axis=0) 
    gfps3[subj,:] = evkd3[0:31].std(axis=0) 
    gfps4[subj,:] = evkd4[0:31].std(axis=0) 
    gfps5[subj,:] = evkd5[0:31].std(axis=0) 
    gfps6[subj,:] = evkd6[0:31].std(axis=0) 
    gfps7[subj,:] = evkd7[0:31].std(axis=0) 
    evkds0[subj,:] = evkd0.mean(axis=0)       #Taking the mean of 4 channels (excluding the EXGs)
    evkds1[subj,:] = evkd1.mean(axis=0)
    evkds2[subj,:] = evkd2.mean(axis=0)
    evkds3[subj,:] = evkd3.mean(axis=0)
    evkds4[subj,:] = evkd4.mean(axis=0)
    evkds5[subj,:] = evkd5.mean(axis=0)
    evkds6[subj,:] = evkd6.mean(axis=0)
    evkds7[subj,:] = evkd7.mean(axis=0)
    
### Sum of squares -- Not doing X-Mean, as 'mean' here is the evoked response here is already referenced to the earlobes
picks_ss = [3, 30, 26, 4, 25, 7, 31, 22, 8, 21, 11, 12, 18]

ss_all12 = np.zeros((len(subjlist),t_full.size))
ss_all20 = np.zeros((len(subjlist),t_full.size))

for subj in range(len(subjlist)):
    sub = subjlist [subj]
    dat = io.loadmat(save_mat_loc + sub + '_1-40Hz_Evoked_AllChan.mat', squeeze_me=True)
    dat.keys()    
    ss6 = dat['evkd6'][picks_ss]      ###Doing this only for the 5 sec plot 
    ss7 = dat['evkd7'][picks_ss]
    ss_all12[subj,:] = (ss6**2).sum(axis=0)
    ss_all20[subj,:] = (ss7**2).sum(axis=0)

#%% Plotting and saving of 32 channel GFP across all subjects -- Baselined for 1 sec interval for coh and incoherent periods

fig, ax = plt.subplots(3, 1, sharex = True, constrained_layout=True)
ax[0].errorbar(t, gfps0.mean(axis=0), yerr=sem(gfps0), label = '12 tone coherence (N='+ str(len(subjlist)) +')', color='green', linewidth=2, ecolor='darkseagreen')
ax[0].errorbar(t, gfps1.mean(axis=0), yerr=sem(gfps1), label = '20 tone coherence (N='+ str(len(subjlist)) +')', color='purple', linewidth=2, ecolor='thistle')
ax[1].errorbar(t, gfps2.mean(axis=0), yerr=sem(gfps2), color='green', linewidth=2, ecolor='darkseagreen')
ax[1].errorbar(t, gfps3.mean(axis=0), yerr=sem(gfps3), color='purple', linewidth=2, ecolor='thistle')
ax[2].errorbar(t, gfps4.mean(axis=0), yerr=sem(gfps4), color='green', linewidth=2, ecolor='darkseagreen')
ax[2].errorbar(t, gfps5.mean(axis=0), yerr=sem(gfps5), color='purple', linewidth=2, ecolor='thistle')
plt.suptitle('Binding - GFP (N='+ str(len(subjlist)) +')')
fig.subplots_adjust(top=0.88)
ax[0].set_title('Onset')
ax[1].title.set_text('Incoherent to Coherent')
ax[2].title.set_text('Coherent to Incoherent')
ax[0].legend()
# fig.tight_layout()
fig.text(0, 0.55,'Global Field Power(\u03bcV)',fontsize=14, va='center', rotation='vertical')
plt.xlabel('Time(s)',fontsize=14)
plt.rcParams["figure.figsize"] = (6.5,5)
plt.xticks(fontsize=14)
plt.show()

### Calculating DC shift from 300-800 ms (GFP)

t1 = t>=0.3
t2 = t<=0.8
t3 = np.array([t2[i] and t1[i] for i in range(len(t1))])

gfp12_coh = gfps2[:,t3]
gfp12_incoh = gfps4[:,t3]
gfp20_coh = gfps3[:,t3]
gfp20_incoh = gfps5[:,t3]

gfp12 = (gfp12_coh - gfp12_incoh).mean(axis=1)         #Calculating coherent - incoherent (kinda baselining)
gfp20 = (gfp20_coh - gfp20_incoh).mean(axis=1)

plt.bar(('12','20'), (gfp12.mean(), gfp20.mean()))
plt.show()

mat_id = dict(sub=subjlist,gfp12 = gfp12, gfp20=gfp20,gfp12_coh=gfp12_coh,gfp12_incoh=gfp12_incoh,
              gfp20_coh=gfp20_coh,gfp20_incoh=gfp20_incoh)

# savemat(save_mat_loc + 'AllSubj_GFPDiff_0.4-40Hz_1sec(N=39).mat', mat_id)

#%% Calculation and saving of 32 channel DC shift in GFP across all subjects -- For entire 5 second duration

plt.errorbar(t_full, gfps6.mean(axis=0), yerr=sem(gfps6), label = '12 tone coherence (N='+ str(len(subjlist)) +')', color='#1b9e77', linewidth=2, ecolor='#1b9e77',alpha=0.3)
plt.errorbar(t_full, gfps7.mean(axis=0), yerr=sem(gfps7), label = '20 tone coherence (N='+ str(len(subjlist)) +')', color='#d95f02', linewidth=2, ecolor='#d95f02',alpha=0.3)
plt.suptitle('Binding (N='+ str(len(subjlist)) +')')
plt.subplots_adjust(top=0.88)
plt.legend()
# fig.tight_layout()
plt.ylabel('Global Field Power(\u03bcV)')
plt.xlabel('Time(s)',fontsize=14)
plt.rcParams["figure.figsize"] = (6.5,5)
plt.xticks(fontsize=14)
plt.show()

### Calculating DC shift from 300-800 ms (GFP)
t1 = t_full>=0.3
t2 = t_full<=0.8
t3 = np.array([t2[i] and t1[i] for i in range(len(t1))])
t4 = t_full>=1.3
t5 = t_full<=1.8
t6 = (np.array([t4[i] and t5[i] for i in range(len(t4))]))
t7 = t_full>=2.3
t8 = t_full<=2.8
t9 = np.array([t7[i] and t8[i] for i in range(len(t7))])
t10 = t_full>=3.3
t11 = t_full<=3.8
t12 = np.array([t10[i] and t11[i] for i in range(len(t10))])
t13 = t_full>=4.3
t14 = t_full<=4.8
t15 = np.array([t13[i] and t14[i] for i in range(len(t13))])

gfp12_1=gfps6[:,t3]
gfp12_2=gfps6[:,t6]
gfp12_3=gfps6[:,t9]
gfp12_4=gfps6[:,t12]
gfp12_5=gfps6[:,t15]

gfp20_1=gfps7[:,t3]
gfp20_2=gfps7[:,t6]
gfp20_3=gfps7[:,t9]
gfp20_4=gfps7[:,t12]
gfp20_5=gfps7[:,t15]

gfp12_coh = gfp12_2 + gfp12_4
gfp12_incoh =  gfp12_3 + gfp12_5
gfps12_5sec=(gfp12_coh-gfp12_incoh).mean(axis=1)

gfp20_coh = gfp20_2 + gfp20_4
gfp20_incoh =  gfp20_3 + gfp20_5
gfps20_5sec=(gfp20_coh-gfp20_incoh).mean(axis=1)

plt.bar(('12','20'), (gfps12_5sec.mean(), gfps20_5sec.mean()))
plt.show()

mat_ids1=dict(subj=subjlist,gfps12_5sec=gfps12_5sec, gfps20_5sec=gfps20_5sec)

# savemat(save_mat_loc + 'AllSubj_GFPDiff_0.4-40Hz_5sec(N=39)_New.mat', mat_ids1)

#%%% #%% Plotting sum of squares across all subjects -- Baselined for 1 sec interval for coh and incoherent periods

plt.errorbar(t_full, ss_all12.mean(axis=0), yerr=sem(ss_all12), label = '12 tone coherence (N='+ str(len(subjlist)) +')', color='green', linewidth=2, ecolor='darkseagreen')
plt.errorbar(t_full, ss_all20.mean(axis=0), yerr=sem(ss_all20), label = '20 tone coherence (N='+ str(len(subjlist)) +')', color='purple', linewidth=2, ecolor='thistle')
plt.suptitle('Binding - Sum of Squares (N='+ str(len(subjlist)) +')')
plt.subplots_adjust(top=0.88)
plt.legend()
# fig.tight_layout()
plt.ylabel('Sum of Squares (\u03bcV)')
plt.xlabel('Time(s)',fontsize=14)
plt.rcParams["figure.figsize"] = (6.5,5)
plt.xticks(fontsize=14)
plt.show()

#%% Plotting and calculating only A32 channel amp DC shift for 300-800 ms for all subjects 
fig, ax = plt.subplots(3, 1, sharex = True, constrained_layout=True)
ax[0].errorbar(t, evkds0.mean(axis=0), yerr=sem(evkds0), label = '12 tone coherence (N='+ str(len(subjlist)) +')', color='green', linewidth=2, ecolor='darkseagreen')
ax[0].errorbar(t, evkds1.mean(axis=0), yerr=sem(evkds1), label = '20 tone coherence (N='+ str(len(subjlist)) +')', color='purple', linewidth=2, ecolor='thistle')
ax[1].errorbar(t, evkds2.mean(axis=0), yerr=sem(evkds2), color='green', linewidth=2, ecolor='darkseagreen')
ax[1].errorbar(t, evkds3.mean(axis=0), yerr=sem(evkds3), color='purple', linewidth=2, ecolor='thistle')
ax[2].errorbar(t, evkds4.mean(axis=0), yerr=sem(evkds4), color='green', linewidth=2, ecolor='darkseagreen')
ax[2].errorbar(t, evkds5.mean(axis=0), yerr=sem(evkds5), color='purple', linewidth=2, ecolor='thistle')
plt.suptitle('Binding A32 - Amplitude (N='+ str(len(subjlist)) +')')
fig.subplots_adjust(top=0.88)
ax[0].set_title('Onset')
ax[1].title.set_text('Incoherent to Coherent')
ax[2].title.set_text('Coherent to Incoherent')
ax[0].legend()
# fig.tight_layout()
fig.text(0, 0.55,'Amplitude (\u03bcV)',fontsize=14, va='center', rotation='vertical')
plt.xlabel('Time(s)',fontsize=14)
plt.rcParams["figure.figsize"] = (6.5,5)
plt.xticks(fontsize=14)
plt.show()

#%%##Calculating P1, P2 peaks and latencies 

tP1_start, tP1_end = 0, 0.15
tP2_start, tP2_end = 0.16, 0.3
t_avg_start, t_avg_end = 0.3, 0.8

binding_results = {}

evkds_all = {'Onset12':evkds0, 
             'Onset20':evkds1, 
             'Incoherent2Coherent_12':evkds2,
             'Coherent2Incoherent_12':evkds3,
             'Incoherent2Coherent_20':evkds4,
             'Coherent2InCoherent_20':evkds5}

# Loop through conditions
for condition, evkds in evkds_all.items():
    P1_peak = []
    P1_latency = []
    P2_peak = []
    P2_latency = []
    avg_steadystate = []
    
    # Iterate through each evoked dataset
    for evkd in evkds:
        # Find indices corresponding to the time slots
        t1_indices = np.where((t >= tP1_start) & (t <= tP1_end))[0]
        t2_indices = np.where((t >= tP2_start) & (t <= tP2_end))[0]
        t_avg_indices = np.where((t >= t_avg_start) & (t <= t_avg_end))[0]
        
        # Extract data within the time slots
        data_t1 = evkd[t1_indices]
        data_t2 = evkd[t2_indices]
        data_avg = evkd[t_avg_indices]
        
        # Find the index corresponding to the peak value in each time slot
        peak_index_t1 = t1_indices[data_t1.argmax()]
        peak_index_t2 = t2_indices[data_t2.argmax()]
        
        # Convert the peak indices to time
        peak_latency_t1 = t[peak_index_t1]
        peak_latency_t2 = t[peak_index_t2]
        
        # Get the peak values
        peak_value_t1 = (data_t1.max())*1e6
        peak_value_t2 = (data_t2.max())*1e6
        
        P1_peak.append(peak_value_t1)
        P1_latency.append(peak_latency_t1)
        P2_peak.append(peak_value_t2)
        P2_latency.append(peak_latency_t2)
        
        # Calculate average response within the average time slot
        avg_response = (np.mean(data_avg))*1e6
        avg_steadystate.append(avg_response)
    
    # Store results in the dictionary
    binding_results[condition] = {'subject':subjlist, 
                                  'P1_peak': P1_peak,
                                  'P1_latency': P1_latency,
                                  'P2_peak': P2_peak,
                                  'P2_latency': P2_latency,
                                  'Avg_steadystate': avg_steadystate}

# savemat(save_mat_loc + 'Binding_1-40Hz_Peaks(N=42).mat', binding_results)

#%%
### Calculating DC shift from 300-800 ms (Amplitude) -- Baselined -- 1 second

t1 = t>=0.3
t2 = t<=0.8
t3 = np.array([t2[i] and t1[i] for i in range(len(t1))])

evoked12_coh = evkds2[:,t3]
evoked12_incoh = evkds4[:,t3]
evoked20_coh = evkds3[:,t3]
evoked20_incoh = evkds5[:,t3]

evoked12 = (evoked12_coh - evoked12_incoh).mean(axis=1)     #Calculating coherent - incoherent (kinda baselining)
evoked20 = (evoked20_coh - evoked20_incoh).mean(axis=1)

plt.bar(('12','20'), (evoked12.mean(), evoked20.mean()))
plt.show()

mat_id = dict(sub=subjlist,evoked12 = evoked12, evoked20=evoked20,evoked12_coh=evoked12_coh,evoked12_incoh=evoked12_incoh,
              evoked20_coh=evoked20_coh,evoked20_incoh=evoked20_incoh)

savemat(save_mat_loc + 'AllSubj_EvokedDiff(4chan)_1-40Hz_1sec(N=39).mat', mat_id)

#%% Plotting and calculating only A32 channel amp DC shift for 300-800 ms for all subjects - Without baseline

plt.errorbar(t_full, evkds6.mean(axis=0), yerr=sem(evkds6), label = '12 tone coherence (N='+ str(len(subjlist)) +')', color='green', linewidth=2, ecolor='darkseagreen')
plt.errorbar(t_full, evkds7.mean(axis=0), yerr=sem(evkds7), label = '20 tone coherence (N='+ str(len(subjlist)) +')', color='purple', linewidth=2, ecolor='thistle')
plt.suptitle('Binding (A32) - Amplitude (N='+ str(len(subjlist)) +')')
plt.subplots_adjust(top=0.88)
plt.legend()
# fig.tight_layout()
plt.ylabel('Amplitude (\u03bcV)')
plt.xlabel('Time(s)',fontsize=14)
plt.rcParams["figure.figsize"] = (6.5,5)
plt.xticks(fontsize=14)
plt.show()

### Calculating DC shift from 300-800 ms (Amplitude for A32 only) -- 5 seconds duration
t1 = t_full>=0.3
t2 = t_full<=0.8
t3 = np.array([t2[i] and t1[i] for i in range(len(t1))])
t4 = t_full>=1.3
t5 = t_full<=1.8
t6 = (np.array([t4[i] and t5[i] for i in range(len(t4))]))
t7 = t_full>=2.3
t8 = t_full<=2.8
t9 = np.array([t7[i] and t8[i] for i in range(len(t7))])
t10 = t_full>=3.3
t11 = t_full<=3.8
t12 = np.array([t10[i] and t11[i] for i in range(len(t10))])
t13 = t_full>=4.3
t14 = t_full<=4.8
t15 = np.array([t13[i] and t14[i] for i in range(len(t13))])
 
evoked12_1=evkds6[:,t3]
evoked12_2=evkds6[:,t6]
evoked12_3=evkds6[:,t9]
evoked12_4=evkds6[:,t12]
evoked12_5=evkds6[:,t15]

evoked20_1=evkds7[:,t3]
evoked20_2=evkds7[:,t6]
evoked20_3=evkds7[:,t9]
evoked20_4=evkds7[:,t12]
evoked20_5=evkds7[:,t15]

evoked12_coh = evoked12_2 + evoked12_4
evoked12_incoh =  evoked12_3 + evoked12_5
evokeds12_5sec=(evoked12_coh-evoked12_incoh).mean(axis=1)

evoked20_coh = evoked20_2 + evoked20_4
evoked20_incoh =  evoked20_3 + evoked20_5
evokeds20_5sec=(evoked20_coh-evoked20_incoh).mean(axis=1)

plt.bar(('12','20'), (evokeds12_5sec.mean(), evokeds20_5sec.mean()))
plt.show()

mat_ids1=dict(subj=subjlist,evokeds12_5sec=evokeds12_5sec, evokeds20_5sec=evokeds20_5sec)

savemat(save_mat_loc + 'AllSubj_A32_EvokedDiff_0.4-40Hz_5sec(N=39).mat', mat_ids1)

