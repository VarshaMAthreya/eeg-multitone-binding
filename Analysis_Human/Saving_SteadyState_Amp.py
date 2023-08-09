# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:23:11 2023

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

plt.switch_backend('QT5Agg')  # Making the plots interactive (Scrollable)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# Defining the dimensions and quality of figures
plt.rcParams["figure.figsize"] = (5.5,5)
plt.rcParams['figure.dpi'] = 120
#%%Setting up stuff
save_loc='C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Human_Analysis/Figures/'
epochs_loc = 'D:/PhD/Data/Epochs-fif/'
save_mat_loc = 'D:/PhD/Data/Binding_matfiles/0.4-40Hz/'

subjlist = ['S268', 'S274', 'S282', 'S285',
            'S277', 'S279', 'S280', 'S259', 'S270', 
            'S271', 'S281', 'S290', 'S284', 'S305',
            'S303', 'S288', 'S260', 'S352', 'S341',
            'S312', 'S347', 'S340', 'S078', 'S342', 'S072', 'S105', 
            'S291', 'S310', 'S339', 'S355',  
            'S272', 'S069', 'S088', 'S309', 'S345']

### Need 'S246','S104', 'S345',
#Removed --  'S273'
#%% Loading all subjects' data 

# 0 - 12 Onset (Upto 1.1 s) -- 0
# 1 - 20 Onset (Upto 1.1 s) -- 1
# 10 - 12 Incoherent to Coherent -- 2
# 11 - 20 Incoherent to Coherent -- 3
# 12 - 12 Coherent to Incoherent -- 4
# 13 - 20 Coherent to Incoherent -- 5
# 14 - 12 Full 5 seconds -- 6
# 15 - 20 Full 5 seconds -- 7 

for subj in range(len(subjlist)):
    sub = subjlist [subj]
    dat = io.loadmat(save_mat_loc + sub + '_0.4-40Hz_Evoked_AllChan.mat', squeeze_me=True)
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
    
for subj in range(len(subjlist)):
    sub = subjlist [subj]
    dat = io.loadmat(save_mat_loc + sub + '_0.4-40Hz_Evoked_AllChan.mat', squeeze_me=True)
    dat.keys()    
    evkd0 = dat['evkd0']
    evkd1 = dat['evkd1']
    evkd2 = dat['evkd2']
    evkd3 = dat['evkd3']
    evkd4 = dat['evkd4']
    evkd5 = dat['evkd5']
    evkd6 = dat['evkd6']
    evkd7 = dat['evkd7']
    gfps0[subj,:] = evkd0[0:31].std(axis=0)     #Taking the SD of the first 32 channels, excluding the EXGs
    gfps1[subj,:] = evkd1[0:31].std(axis=0) 
    gfps2[subj,:] = evkd2[0:31].std(axis=0) 
    gfps3[subj,:] = evkd3[0:31].std(axis=0) 
    gfps4[subj,:] = evkd4[0:31].std(axis=0) 
    gfps5[subj,:] = evkd5[0:31].std(axis=0) 
    gfps6[subj,:] = evkd6[0:31].std(axis=0) 
    gfps7[subj,:] = evkd7[0:31].std(axis=0) 
    evkds0[subj,:] = evkd0[0:31].mean(axis=0)       #Taking the mean of only 32 channels (excluding the EXGs)
    evkds1[subj,:] = evkd1[0:31].mean(axis=0)
    evkds2[subj,:] = evkd2[0:31].mean(axis=0)
    evkds3[subj,:] = evkd3[0:31].mean(axis=0)
    evkds4[subj,:] = evkd4[0:31].mean(axis=0)
    evkds5[subj,:] = evkd5[0:31].mean(axis=0)
    evkds6[subj,:] = evkd6[0:31].mean(axis=0)
    evkds7[subj,:] = evkd7[0:31].mean(axis=0)

#%% Loading S337's data -- as it is without the EXGs, so the chans saved are only 32 



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
fig.tight_layout()
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

gfp12 = gfp12_coh - gfp12_incoh         #Calculating coherent - incoherent (kinda baselining)
gfp20 = gfp20_coh - gfp20_incoh

mat_id = dict(subj=subj,gfp12 = gfp12, gfp20=gfp20,gfp12_coh=gfp12_coh,gfp12_incoh=gfp12_incoh,
              gfp20_coh=gfp20_coh,gfp20_incoh=gfp20_incoh)

savemat(save_mat_loc + 'AllSubj_GFPDiff.mat', mat_id)

#%% Calculation and saving of 32 channel DC shift in GFP across all subjects -- Baselined for 1 sec interval for coh and incoherent periods





#%% Calculating only A32 channel amp DC shift for 300-800 ms for all subjects 
