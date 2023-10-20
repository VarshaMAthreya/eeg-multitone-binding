# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 13:41:14 2023

@author: vmysorea
"""
import sys
sys.path.append('C:/Users/vmysorea/Documents/mne-python/')
import warnings
import mne
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
save_mat_loc = 'D:/PhD/Data/Binding_matfiles/1-40Hz/'

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

subjlist_o = ['S312', 'S347', 'S340','S078','S069', 
              'S088','S342','S072','S308','S344','S105',
              'S291','S310','S339']


#S104 and S345 excluded (weird data) - S337 no EXGs
#%% Loading files for 12 and 20 condition for all subjects - No age separation 
evokeds_12 = []
evokeds_20 =[]

evk12 = np.zeros((len(subjlist),22529))
evk20 = np.zeros((len(subjlist),22529))
gfps12 = np.zeros((len(subjlist),22529))
gfps20 = np.zeros((len(subjlist),22529))

for subj in range(len(subjlist)):
    sub = subjlist [subj]
    dat = io.loadmat(save_mat_loc + sub + '_1-40Hz_Evoked_AllChan.mat', squeeze_me=True)
    dat.keys()
    evoked = dat['evoked']
    evoked12 = dat['evoked12']
    evoked20 = dat['evoked20']
    fs = dat['fs']
    t = dat['t']  
    gfp12=evoked12.std(axis=0)   
    gfps12[subj,:]=gfp12
    gfp20=evoked20.std(axis=0)   
    gfps20[subj,:]=gfp20
    evokeds_12 += [evoked12,]
    y = evoked12.mean(axis=0)
    evk12[subj,:] = y
    evokeds_20 += [evoked20,]
    x = evoked20.mean(axis=0)
    evk20[subj,:] = x

evk12_avg = evk12.mean(axis=0)*1e6
evk20_avg = evk20.mean(axis=0)*1e6
evk12_sem = sem(evk12)*1e6
evk20_sem = sem(evk20)*1e6
xmax=max(evk20_avg)

gfps12_avg = gfps12.mean(axis=0)*1e6
gfps20_avg = gfps20.mean(axis=0)*1e6
gfps12_sem = sem(gfps12)*1e6
gfps20_sem = sem(gfps20)*1e6
ymax=max(gfps20_avg)

##Plotting GFPs 
fig, ax = plt.subplots(constrained_layout=True)
plt.errorbar(t, gfps12_avg, yerr=gfps12_sem,  label = '12 Tone Coherence', color='green', linewidth=2, ecolor='palegreen')
plt.errorbar(t, gfps20_avg, yerr=gfps20_sem, label = '20 Tone Coherence', color='purple', linewidth=2, ecolor='thistle')
plt.title('Binding-GFP (N=36)')
plt.vlines(x=[0,1,2,3,4,5], ymin=0, ymax= ymax+0.4, colors='black', ls='--')
ax.text(0, ymax+0.4, 'Stim On', va='center', ha='center', fontsize = 12)
ax.text(0.5, 0.5, 'Incoherent', va='center', ha='center', fontsize = 11)
ax.text(1.5, 0.5, 'Coherent', va='center', ha='center', fontsize = 11)
ax.text(2.5, 0.5, 'Incoherent', va='center', ha='center', fontsize = 11)
ax.text(3.5, 0.5, 'Coherent', va='center', ha='center', fontsize = 11)
ax.text(4.5, 0.5, 'Incoherent', va='center', ha='center', fontsize = 11)
ax.text(5, ymax+0.4, 'Stim End', va='center', ha='center', fontsize = 12)
ax.axvspan(1.3,1.8, alpha=0.3,color='lightgrey')
ax.axvspan(3.3,3.8, alpha=0.3,color='lightgrey')
plt.xlabel('Time(s)',fontsize=20)
plt.ylabel('Global Field Power(\u03bcV)',fontsize=20)
# plt.tight_layout()
plt.rcParams["figure.figsize"] = (6.5,5)
plt.ylim(0,ymax+0.35)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper right',fontsize='medium')
plt.show()

plt.savefig(save_loc + 'GFP_with12', dpi=500)

##Plotting evoked responses
fig1, ax = plt.subplots(constrained_layout=True)
plt.errorbar(t, evk12_avg, yerr=evk12_sem,  label = '12 Tone Coherence', color='green', linewidth=2, ecolor='palegreen')
plt.errorbar(t, evk20_avg, yerr=evk20_sem, label = '20 Tone Coherence', color='purple', linewidth=2, ecolor='thistle')
plt.vlines(x=[0,1,2,3,4,5], ymin=-0.5, ymax= xmax+0.5, colors='black', ls='--')
ax.text(0, xmax+0.35, 'Stim On', va='center', ha='center', fontsize = 12)
ax.text(0.5, -0.45, 'Incoherent', va='center', ha='center', fontsize = 11)
ax.text(1.5, -0.45, 'Coherent', va='center', ha='center', fontsize = 11)
ax.text(2.5, -0.45, 'Incoherent', va='center', ha='center', fontsize = 11)
ax.text(3.5, -0.45, 'Coherent', va='center', ha='center', fontsize = 11)
ax.text(4.5, -0.45, 'Incoherent', va='center', ha='center', fontsize = 11)
ax.text(5, xmax+0.35, 'Stim End', va='center', ha='center', fontsize = 12)
ax.axvspan(1.3,1.8, alpha=0.3,color='lightgrey')
ax.axvspan(3.3,3.8, alpha=0.3,color='lightgrey')
plt.ylim(-0.5,1.8)
plt.xlabel('Time(s)', fontsize=14)
plt.ylabel('Amplitude(\u03bcV)', fontsize=14)
plt.title('Binding - Evoked response (N=36)')
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
plt.legend(fontsize='medium')
plt.rcParams["figure.figsize"] = (6.5,5)
plt.show()


plt.savefig(save_loc + 'Evoked_with12', dpi=500)

#%%Loading mat files for 20 condition only and plotting outside MNE for GFP 
evokeds_y = []
evokeds_o =[]

gfps20_y = np.zeros((len(subjlist_y),22529))
evk_y = np.zeros((len(subjlist_y),22529))

for subj in range(len(subjlist_y)):
    sub = subjlist_y[subj]
    dat = io.loadmat(save_mat_loc + sub + '_allevoked0.4_with12.mat', squeeze_me=True)
    dat.keys()
    evoked_y = dat['evoked']
    evoked20_y = dat['evoked20']
    fs = dat['fs']
    t = dat['t']  
    gfp20_y=evoked20_y.std(axis=0)   
    gfps20_y[subj,:]=gfp20_y
    evokeds_y += [evoked20_y,]
    y = evoked20_y.mean(axis=0)
    evk_y[subj,:] = y

gfps20_o = np.zeros((len(subjlist_o),22529))
evk_o = np.zeros((len(subjlist_o),22529))
for subo in range(len(subjlist_o)):
    subs = subjlist_o[subo]
    dat = io.loadmat(save_mat_loc + subs + '_allevoked0.4_with12.mat', squeeze_me=True)
    dat.keys()
    evoked_o = dat['evoked']
    evoked20_o = dat['evoked20']
    fs = dat['fs']
    t = dat['t']   
    gfp20_o=evoked20_o.std(axis=0)   
    gfps20_o[subo,:]=gfp20_o 
    evokeds_o += [evoked20_o,]
    x = evoked20_o.mean(axis=0)
    evk_o[subo,:] = x
 
a = gfps20_y.mean(axis=0)*1e6
b = gfps20_o.mean(axis=0)*1e6
r = sem(gfps20_y)*1e6
s = sem(gfps20_o)*1e6
ymax=max(a)

##Plotting GFPs 
fig, ax = plt.subplots(constrained_layout=True)
plt.errorbar(t, a, yerr=r,  label = 'Below 35 y (N='+ str(len(subjlist_y)) +')', color='green', linewidth=2, ecolor='darkseagreen')
plt.errorbar(t, b, yerr=s, label = 'Above 35 y (N=15)', color='purple', linewidth=2, ecolor='thistle')
# plt.errorbar(t, a, yerr=r,  label = 'Below 35 y (N=' + str(len(subjlist_y)) + ')', color='green', linewidth=2, ecolor='darkseagreen')
# plt.errorbar(t, b, yerr=s, label = 'Above 35 y (N=' + str(len(subjlist_o)) + ')', color='purple', linewidth=2, ecolor='thistle')
plt.title('Binding - GFP (N=36)')
plt.vlines(x=[0,1,2,3,4,5], ymin=0, ymax= ymax+0.4, colors='black', ls='--')
ax.text(0, ymax+0.4, 'Stim On', va='center', ha='center', fontsize = 12)
ax.text(0.5, 0.5, 'Incoherent', va='center', ha='center', fontsize = 11)
ax.text(1.5, 0.5, 'Coherent', va='center', ha='center', fontsize = 11)
ax.text(2.5, 0.5, 'Incoherent', va='center', ha='center', fontsize = 11)
ax.text(3.5, 0.5, 'Coherent', va='center', ha='center', fontsize = 11)
ax.text(4.5, 0.5, 'Incoherent', va='center', ha='center', fontsize = 11)
ax.text(5, ymax+0.4, 'Stim End', va='center', ha='center', fontsize = 12)
ax.axvspan(1.3,1.8, alpha=0.3,color='lightgrey')
ax.axvspan(3.3,3.8, alpha=0.3,color='lightgrey')
plt.xlabel('Time(s)',fontsize=20)
plt.ylabel('Global Field Power(\u03bcV)',fontsize=20)
# plt.tight_layout()
plt.rcParams["figure.figsize"] = (6.5,5)
plt.ylim(0,ymax+0.35)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper right',fontsize='medium')
plt.show()

plt.savefig(save_loc + 'YoungvsOld_GFP_New', dpi=500)

###Plotting evoked responses 
c = evk_y.mean(axis=0)*1e6
d = evk_o.mean(axis=0)*1e6
e = sem(evk_y)*1e6
f = sem(evk_o)*1e6

##Plotting evokeds
fig1, ax = plt.subplots(constrained_layout=True)
plt.errorbar(t, c, yerr=e,  label = 'Below 35 y (N=21)', color='green', linewidth=2, ecolor='palegreen')
plt.errorbar(t, d, yerr=f, label = 'Above 35 y (N=15)', color='purple', linewidth=2, ecolor='thistle')
# plt.errorbar(t, c, yerr=e,  label = 'Below 35 y (N=' + str(len(subjlist_y)) + ')', color='green', linewidth=2, ecolor='palegreen')
# plt.errorbar(t, d, yerr=f, label = 'Above 35 y (N=' + str(len(subjlist_o)) + ')', color='purple', linewidth=2, ecolor='thistle')
# plt.vlines(x=[0,1,2,3,4,5], ymin=-0.5, ymax= ymax, colors='black', ls='--')
# plt.ylim(-0.5,2)
plt.xlabel('Time(s)')
plt.ylabel('Amplitude(\u03bcV)')
plt.title('Binding - Evoked response (N=36)')
plt.legend(fontsize='x-small')
plt.rcParams["figure.figsize"] = (6,5)
plt.show()
plt.savefig(save_loc + 'YoungvsOld_Evoked', dpi=500)

#%%Saving mats for plotting shifts from baseline
t1 = t>=0.3
t2 = t<=0.8
t3 = np.array([t2[i] and t1[i] for i in range(len(t1))])
t4 = t>=1.3
t5 = t<=1.8
t6 = (np.array([t4[i] and t5[i] for i in range(len(t4))]))
t7 = t>=2.3
t8 = t<=2.8
t9 = np.array([t7[i] and t8[i] for i in range(len(t7))])
t10 = t>=3.3
t11 = t<=3.8
t12 = np.array([t10[i] and t11[i] for i in range(len(t10))])
t13 = t>=4.3
t14 = t<=4.8
t15 = np.array([t13[i] and t14[i] for i in range(len(t13))])

gfp_1_y=np.zeros((len(subjlist_y),len(t3)))
gfp_2_y=np.zeros((len(subjlist_y),len(t6)))
gfp_3_y=np.zeros((len(subjlist_y),len(t9)))
gfp_4_y=np.zeros((len(subjlist_y),len(t12)))
gfp_5_y=np.zeros((len(subjlist_y),len(t15)))

gfp_1_y=gfps20_y[:,t3]
gfp_2_y=gfps20_y[:,t6]
gfp_3_y=gfps20_y[:,t9]
gfp_4_y=gfps20_y[:,t12]
gfp_5_y=gfps20_y[:,t15]

gfp_coh_y = (gfp_2_y + gfp_4_y).mean(axis=1)
gfp_incoh_y=  gfp_3_y + gfp_5_y
gfps_y=(gfp_coh_y-gfp_incoh_y).mean(axis=1)

gfp_1_o=np.zeros((len(subjlist_o),len(t3)))
gfp_2_o=np.zeros((len(subjlist_o),len(t6)))
gfp_3_o=np.zeros((len(subjlist_o),len(t9)))
gfp_4_o=np.zeros((len(subjlist_o),len(t12)))
gfp_5_o=np.zeros((len(subjlist_o),len(t15)))

gfp_1_o=gfps20_o[:,t3]
gfp_2_o=gfps20_o[:,t6]
gfp_3_o=gfps20_o[:,t9]
gfp_4_o=gfps20_o[:,t12]
gfp_5_o=gfps20_o[:,t15]

gfp_coh_o = (gfp_2_o + gfp_4_o).mean(axis=1)
gfp_incoh_o=  gfp_3_o + gfp_5_o
gfps_o=(gfp_coh_o-gfp_incoh_o).mean(axis=1)

mat_ids1=dict(subj=subj,gfp_coh_y=gfp_coh_y)
mat_ids2=dict(subj=subo,gfp_coh_o=gfp_coh_o)

savemat(save_mat_loc + 'Young_GFPs_CohOnly4.3.mat', mat_ids1)
savemat(save_mat_loc + 'Old_GFPs_CohOnly4.3.mat', mat_ids2)

plt.bar(('Young', 'Old'),((gfps_y.mean(axis=0)*1e6),(gfps_o.mean(axis=0)*1e6)))
plt.show()

################################################################
evokeds_y = []
evokeds_o = []
evokeds20_y = []
evokeds12_y = []
evokeds20_o = []
evokeds12_o = []

for subj in subjlist_y:
    evoked_y = mne.read_evokeds(epochs_loc + subj +'_Binding_evoked-ave.fif',baseline=(-0.3, 0), proj=True)
    evoked20_y = mne.read_evokeds(epochs_loc + subj +'_Binding_evoked20-ave.fif',baseline=(-0.3, 0), proj=True)
    evoked12_y = mne.read_evokeds(epochs_loc + subj +'_Binding_evoked12-ave.fif',baseline=(-0.3, 0), proj=True)
    # mapping[subj] = [evoked20, evoked12,evoked] 
    evokeds_y.append(evoked_y)
    evokeds20_y+=[evoked20_y,]
    evokeds12_y+=[evoked12_y,]
    
    #Onset_ftime = evoked.plot(spatial_colors=False,  gfp=True, exclude = ['A1','A2', 'A30', 'A16'])   #Butterfly plots

# for subj in subjlist_m:
#     evoked_m = mne.read_evokeds(epochs_loc + subj +'_Binding_evoked-ave.fif',baseline=(-0.3, 0), proj=True)
#     evoked20_m = mne.read_evokeds(epochs_loc + subj +'_Binding_evoked20-ave.fif',baseline=(-0.3, 0), proj=True)
#     evoked12_m = mne.read_evokeds(epochs_loc + subj +'_Binding_evoked12-ave.fif',baseline=(-0.3, 0), proj=True)
#     # mapping[subj] = [evoked20, evoked12,evoked] 
#     evokeds_m+=[evoked_m,]
#     evokeds20_m+=[evoked20_m,]
#     evokeds12_m+=[evoked12_m,]
    
for subj in subjlist_o:
    evoked_o = mne.read_evokeds(epochs_loc + subj +'_Binding_evoked-ave.fif',baseline=(-0.3, 0), proj=True)
    evoked20_o = mne.read_evokeds(epochs_loc + subj +'_Binding_evoked20-ave.fif',baseline=(-0.3, 0), proj=True)
    evoked12_o = mne.read_evokeds(epochs_loc + subj +'_Binding_evoked12-ave.fif',baseline=(-0.3, 0), proj=True)
    # mapping[subj] = [evoked20, evoked12,evoked] 
    evokeds_o+=[evoked_o,]
    evokeds20_o+=[evoked20_o,]
    evokeds12_o+=[evoked12_o,]

for c in range(len(subjlist_y)):
    evokeds_y_all = mne.combine_evoked(evokeds_y[c], weights='equal')
evokeds_y_all.plot(gfp=True, titles=('Below 35 years (N=' + str(len(subjlist_y)) +')'))

for c in range(len(subjlist_y)):
    evokeds20_y_all = mne.combine_evoked(evokeds20_y[c], weights='equal')
evokeds20_y_all.plot(gfp=True, titles=('Below 35 years - 20 tone (N=' + str(len(subjlist_y)) +')'))
    
# evokeds_m_all = mne.combine_evoked(evokeds_m[0], weights='equal')
# evokeds_m_all.plot(gfp='only')

for c in range(len(subjlist_o)):
    evokeds_o_all = mne.combine_evoked(evokeds_o[c], weights='equal')
evokeds_o_all.plot(gfp=True, titles=('Above 35 years (N=' + str(len(subjlist_o)) +')'))

for c in range(len(subjlist_o)):
    evokeds20_o_all = mne.combine_evoked(evokeds20_o[c], weights='equal')
evokeds20_o_all.plot(gfp=True, titles=('Above 35 years - 20 tone (N=' + str(len(subjlist_o)) +')'))

evokeds = [evokeds_y_all, evokeds_o_all]
mne.viz.plot_compare_evokeds(dict(Below35y=evokeds_y_all, Above35y=evokeds_o_all), colors=('green', 'purple'), ci=True,
                             ylim=dict(eeg=[0, 3]),legend='upper right', vlines=[0.,1.,2.,3.,4.],
                             title='Binding Across Age groups')
plt.savefig(save_loc + 'GFP_AcrossAge.png', dpi=300)

# evokeds = mne.grand_average(evokeds_m)

# for c in range(len(subjlist_y)):
#     evokedsy_all = mne.grand_average(evokeds_y[c])
    
# evokeds12 = mne.grand_average(evokeds12[0])

# evokeds.plot(picks=['A31', 'A32'])
# times=(0.2, 0.6,1.3,1.6,2.2,2.6,3.1,3.6,4.2,4.6)
# evokeds_y.plot_topomap(times=times,average=0.4, contours=10, res=32, colorbar=True)
# evokeds.plot_joint(times, title= 'Evoked Response for Binding Stimulus')

# evokeds20.plot_topomap(times=times,average=0.4, contours=10, res=32, colorbar=True)
# evokeds20.plot_joint(times, title= 'Evoked Response for Binding Stimulus (20 tone)')

# evokeds12.plot_topomap(times=times,average=0.4, contours=10, res=32, colorbar=True)
# evokeds12.plot_joint(times, title= 'Evoked Response for Binding Stimulus (12 tone)') 

# evokeds20.plot(gfp='only')  

# gfp_sound_X = evoked_sound_X.data.std(axis=0, ddof=0) #To get GFP 
    
