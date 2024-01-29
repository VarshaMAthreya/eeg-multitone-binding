# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:04:59 2023

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
import pandas as pd
import seaborn as sns
import mne
from mne.channels.layout import _pol_to_cart, _cart_to_sph

plt.switch_backend('QT5Agg')  # Making the plots interactive (Scrollable)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# Defining the dimensions and quality of figures
plt.rcParams["figure.figsize"] = (5.5,5)
plt.rcParams['figure.dpi'] = 500
plt.rc('xtick', labelsize=11) 
plt.rc('ytick', labelsize=11) 

#%%Setting up stuff
fig_loc = 'C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/MTB_Analysis/FinalThesis/ForPPT/'
save_mat_loc = 'D:/PhD/Data/Binding_matfiles/1-40Hz/'
data_loc = 'C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/MTB_Analysis/'

subjlist_y = ['S273','S268','S269','S274','S282',
              'S285','S272','S259','S277','S279',
              'S280','S270','S271','S281','S290',
              'S284','S305','S303','S288','S260',
              'S309','S288','S341','S352','S312',
              'S347','S340']

subjlist_o = ['S078','S069', 'S355','S088','S342',
              'S072','S358','S308','S344','S105',
              'S345','S291','S337','S310','S339']

evokeds_y = []
evokeds_o =[]

for subj in range(len(subjlist_y)):
    suby = subjlist_y[subj]
    dat = io.loadmat(save_mat_loc + suby + '_1-40Hz_Evoked_ALlChan.mat', squeeze_me=True)
    dat.keys()
    evoked_y = (dat['evkd7'][0:32])       #Only 20 Tone Coherence 
    t = dat['t_full']    
    evokeds_y += [evoked_y,]
    y = evoked_y.mean(axis=0)
    
for subj in range(len(subjlist_o)):
    subo = subjlist_o[subj]
    dat = io.loadmat(save_mat_loc + subo + '_1-40Hz_Evoked_ALlChan.mat', squeeze_me=True)
    dat.keys()
    evoked_o = (dat['evkd7'][0:32])       #Only 20 Tone Coherence 
    t = dat['t_full']    
    evokeds_o += [evoked_o,]
    x = evoked_o.mean(axis=0)
  
#%% Montage Setting 

montage = mne.channels.make_standard_montage('biosemi32')

chs = montage._get_ch_pos()
ch_names, xyz = zip(*[(ich, ixyz) for ich, ixyz in chs.items()])
xyz = np.vstack(xyz)

mne.viz.plot_montage(montage)
sph = _cart_to_sph(xyz)
xy = _pol_to_cart(sph[:, 1:][:, ::-1]) * 0.05

#%%Noting down times to calculate topomaps -- Onsets, responses to transitions, steady-state, and offset

t1 = t>=0.0
t2 = t<=0.29
t3 = np.array([t2[i] and t1[i] for i in range(len(t1))])
t4 = t>=0.3
t5 = t<=0.8
t6 = (np.array([t4[i] and t5[i] for i in range(len(t4))]))
t7 = t>=1.0
t8 = t<=1.29
t9 = np.array([t7[i] and t8[i] for i in range(len(t7))])
t10 = t>=1.3
t11 = t<=1.8
t12 = np.array([t10[i] and t11[i] for i in range(len(t10))])
t13 = t>=2.0
t14 = t<=2.29
t15 = np.array([t13[i] and t14[i] for i in range(len(t13))])
t16 = t>=2.3
t17 = t<=2.8
t18 = np.array([t16[i] and t17[i] for i in range(len(t16))])
t19 = t>=3.0
t20 = t<=3.29
t21 = np.array([t19[i] and t20[i] for i in range(len(t19))])
t22 = t>=3.3
t23 = t<=3.8
t24 = np.array([t22[i] and t23[i] for i in range(len(t22))])
t25 = t>=4.0
t26 = t<=4.29
t27 = np.array([t25[i] and t26[i] for i in range(len(t25))])
t28 = t>=4.3
t29 = t<=4.8
t30 = np.array([t28[i] and t29[i] for i in range(len(t28))])
t31 = t>=5.0
t32 = t<=5.29
t33 = np.array([t31[i] and t32[i] for i in range(len(t31))])

  # Young group   
evoked_1=np.zeros((32,len(t3)))     #Onset
evoked_2=np.zeros((32,len(t6)))     #Onset SS
evoked_3=np.zeros((32,len(t9)))     #Incoh 2 Coh 1
evoked_4=np.zeros((32,len(t12)))    #Incoh 2 Coh SS 1
evoked_5=np.zeros((32,len(t15)))    #Coh 2 Incoh 1
evoked_6=np.zeros((32,len(t18)))    #Coh 2 Incoh SS 1
evoked_7=np.zeros((32,len(t21)))    #Incoh 2 Coh 2
evoked_8=np.zeros((32,len(t24)))    #Incoh 2 Coh SS 2
evoked_9=np.zeros((32,len(t27)))    #Coh 2 Incoh 2
evoked_10=np.zeros((32,len(t30)))   #Coh 2 Incoh SS 2
evoked_11=np.zeros((32,len(t33)))   #Offset 

for c in range(len(subjlist_y)):
    evoked_1=evokeds_y[c][:,t3]
    evoked_2=evokeds_y[c][:,t6]
    evoked_3=evokeds_y[c][:,t9]
    evoked_4=evokeds_y[c][:,t12]
    evoked_5=evokeds_y[c][:,t15]
    evoked_6=evokeds_y[c][:,t18]
    evoked_7=evokeds_y[c][:,t21]
    evoked_8=evokeds_y[c][:,t24]
    evoked_9=evokeds_y[c][:,t27]
    evoked_10=evokeds_y[c][:,t30]
    evoked_11=evokeds_y[c][:,t33]
    
#Old group     
o_evoked_1=np.zeros((32,len(t3)))
o_evoked_2=np.zeros((32,len(t6)))
o_evoked_3=np.zeros((32,len(t9)))
o_evoked_4=np.zeros((32,len(t12)))
o_evoked_5=np.zeros((32,len(t15)))
o_evoked_6=np.zeros((32,len(t18)))
o_evoked_7=np.zeros((32,len(t21)))
o_evoked_8=np.zeros((32,len(t24)))
o_evoked_9=np.zeros((32,len(t27)))
o_evoked_10=np.zeros((32,len(t30)))
o_evoked_11=np.zeros((32,len(t33)))

for c in range(len(subjlist_o)):
    o_evoked_1=evokeds_o[c][:,t3]
    o_evoked_2=evokeds_o[c][:,t6]
    o_evoked_3=evokeds_o[c][:,t9]
    o_evoked_4=evokeds_o[c][:,t12]
    o_evoked_5=evokeds_o[c][:,t15]
    o_evoked_6=evokeds_o[c][:,t18]
    o_evoked_7=evokeds_o[c][:,t21]
    o_evoked_8=evokeds_o[c][:,t24]
    o_evoked_9=evokeds_o[c][:,t27]
    o_evoked_10=evokeds_o[c][:,t30]
    o_evoked_11=evokeds_o[c][:,t33]

#Combined stuff 
coh2incoh_y = evoked_5 + evoked_9 
coh2incoh_ss_y = evoked_6 + evoked_10
incoh2coh_y = evoked_3 + evoked_7
incoh2coh_ss_y = evoked_4 + evoked_8
ss_diff_y = coh2incoh_ss_y - incoh2coh_ss_y 

coh2incoh_o = o_evoked_5 + o_evoked_9 
coh2incoh_ss_o = o_evoked_6 + o_evoked_10
incoh2coh_o = o_evoked_3 + o_evoked_7
incoh2coh_ss_o = o_evoked_4 + o_evoked_8
ss_diff_o = coh2incoh_ss_o - incoh2coh_ss_o 

# Replace this with difference calculated from 5-second evoked response
array_to_plot_as_topomap = o_evoked_11.mean(axis=1)

fig,ax = plt.subplots(figsize=(1.5,1.5))
im,_   = mne.viz.plot_topomap(array_to_plot_as_topomap, xy,  contours=15, res=128, size=2, vlim=(-2*1e-6,2*1e-6), 
                              axes=ax,show=False)   
# fig=mne.viz.plot_topomap(array_to_plot_as_topomap, xy, contours=15, res=128, size=2, vlim=(-2*1e-6,2*1e-6))

# cbar = plt.colorbar(ax=ax, orientation='vertical', mappable=im)
# cbar.ax.tick_params(labelsize=5)

# plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.88)
plt.tight_layout()
plt.rcParams['axes.titlepad'] = 0
plt.title('Offset | Old', fontsize=8,loc='center')

plt.savefig(fig_loc + 'Offset_Old.png', dpi=500, transparent=True)

##Saving mat files if I want to analyze SS difference stats later 



#%%Come back to this later 
# for subj in range(len(subjlist)):
#     sub = subjlist [subj]
#     dat1 = io.loadmat(save_mat_loc + sub + '_1-40Hz_Evoked_AllChan.mat', squeeze_me=True)
#     dat1.keys()
#     t = dat1['t']
#     t_full = dat1['t_full']

# #Load data with subjects' age and gender and pick subj only from subjlist 
# dat = pd.read_csv(data_loc + 'subj_age_gender.csv')
# dat0 = dat['Subject'].isin(subjlist)
# dat = dat[dat0]

# #Categorizing into different age groups 
# # def group_age (age):
# #     if age <= 35:
# #         return 'YNH'
# #     elif age <=55:
# #         return 'MNH'
# #     else: 
# #         return 'ONH'
    
# def group_age(age):
#     if age <= 40:
#         return 'YNH'
#     else:
#         return 'ONH'

# dat['age_group'] = dat['Age'].apply([group_age])
# age_groups = dat.groupby(['age_group'], sort=False)

# #%% Loading all subjects' data 

# # 0 - 12 Onset (Upto 1.1 s) -- 0
# # 1 - 20 Onset (Upto 1.1 s) -- 1
# # 10 - 12 Incoherent to Coherent -- 2
# # 11 - 12 Coherent to Incoherent -- 3
# # 12 - 20 Incoherent to Coherent -- 4
# # 13 - 20 Coherent to Incoherent -- 5
# # 14 - 12 Full 5 seconds -- 6
# # 15 - 20 Full 5 seconds -- 7 

# # Initialize empty lists to store data for different conditions and age groups
# evkds0_all = []
# evkds1_all = []
# evkds2_all = []
# evkds3_all = []
# evkds4_all = []
# evkds5_all = []
# evkds6_all = []
# evkds7_all = []
# # ss_all12 =[]
# # ss_all20 =[]
# # gfp_all12 = []
# # gfp_all20 = []

# mat_agegroups = []
# picks = [4, 25, 30, 31]         #Took off the Cs

# # picks_ss = [3, 30, 26, 4, 25, 7, 31, 22, 8, 21, 11, 12, 18]

# for age, groups in age_groups:
#     group_evkds0 = []  # Initialize lists for each condition and age group
#     group_evkds1 = []
#     group_evkds2 = []
#     group_evkds3 = []
#     group_evkds4 = []
#     group_evkds5 = []
#     group_evkds6 = []
#     group_evkds7 = []
#     # group_ss12 =[]
#     # group_ss20 =[]
#     # group_gfp12 =[]
#     # group_gfp20 = []
    
#     for index, column in groups.iterrows():
#         subj = column['Subject']
#         dat = io.loadmat(save_mat_loc + subj + '_1-40Hz_Evoked_AllChan.mat', squeeze_me=True)
#         dat.keys()
#         evkd0 = dat['evkd0'][0:32]
#         evkd1 = dat['evkd1'][0:32]
#         evkd2 = dat['evkd2'][0:32]
#         evkd3 = dat['evkd3'][0:32]
#         evkd4 = dat['evkd4'][0:32]
#         evkd5 = dat['evkd5'][0:32]
#         evkd6 = dat['evkd6'][0:32]
#         evkd7 = dat['evkd7'][0:32]
        
#         # ss6 = dat['evkd6'][picks_ss]      ###Doing this only for the 5 sec plot 
#         # ss7 = dat['evkd7'][picks_ss]
#         # ss12 = ((ss6**2).sum(axis=0))
#         # ss20 = ((ss7**2).sum(axis=0))
        
#         # gfp12 = ss6.std(axis=0)
#         # gfp20 = ss7.std(axis=0)

#         group_evkds0.append(evkd0.mean(axis=0))
#         group_evkds1.append(evkd1.mean(axis=0))
#         group_evkds2.append(evkd2.mean(axis=0))
#         group_evkds3.append(evkd3.mean(axis=0))
#         group_evkds4.append(evkd4.mean(axis=0))
#         group_evkds5.append(evkd5.mean(axis=0))
#         group_evkds6.append(evkd6.mean(axis=0))
#         group_evkds7.append(evkd7.mean(axis=0))
        
#         # group_ss12.append(ss12)
#         # group_ss20.append(ss20)
        
#         # group_gfp12.append(gfp12)
#         # group_gfp20.append(gfp20)
        
#     # Append data for each age group to lists
#     evkds0_all.append(group_evkds0)
#     evkds1_all.append(group_evkds1)
#     evkds2_all.append(group_evkds2)
#     evkds3_all.append(group_evkds3)
#     evkds4_all.append(group_evkds4)
#     evkds5_all.append(group_evkds5)
#     evkds6_all.append(group_evkds6)
#     evkds7_all.append(group_evkds7)
    
# #%% Setting biosemi 32 channel cap settings 

# montage = mne.channels.make_standard_montage('biosemi32')

# chs = montage._get_ch_pos()
# ch_names, xyz = zip(*[(ich, ixyz) for ich, ixyz in chs.items()])
# xyz = np.vstack(xyz)

# mne.viz.plot_montage(montage)
# sph = _cart_to_sph(xyz)
# xy = _pol_to_cart(sph[:, 1:][:, ::-1]) * 0.05

# #%%## Calculating times for response, steady-state for onset, incoherent-coherent and coherent-incoherent transitions 

# intervals = [(0, 0.29), (0.3, 0.8), (1.0, 1.29), (1.3, 1.8), (2.0, 2.29), (2.3, 2.8), (3.0, 3.29), (3.3, 3.8), (4.0, 4.29), (4.3,4.8), (5.0,5.29)]

# t_values = []
# for start, end in intervals:
#     t_values.append((t_full >= start) & (t_full <= end))
    


# # conditions = {6: evkds6_all,
# #               7: evkds7_all}

# # evoked12 = []
# # evoked20 = []
# # for a in t_values:
# #     for subjlist in conditions.values():
# #         evoked_subj12 = [arr[a] for arr in subjlist]
# #         evoked12.append(evoked_subj12)
# #     evokeds12.append(evoked12)
# #     evokeds20.append(evoked20)
    

# #Getting the evoked responses for the above specified times  
# evokeds12 = []
# evokeds20 = []

# for i in t_values:
#     evoked12 = []
#     for subjlist in evkds6_all:
#         evoked_subj = []
#         for arr in subjlist:
#             evoked_subj.append(arr[i])
#         evoked12.append(evoked_subj)
#     evokeds12.append(evoked12)
    
# for i in t_values:
#     evoked20 = []
#     for subjlist in evkds7_all:
#         evoked_subj = []
#         for arr in subjlist:
#             evoked_subj.append(arr[i])
#         evoked20.append(evoked_subj)
#     evokeds20.append(evoked20)
    
# #%% Calculating the mean and SEM within each age group for each condition 
# conditions = {
#     0: evokeds12,
#     1: evokeds20}

# mean_data = {}

# for condition, evkds_all in conditions.items():
#     mean_age_group = []
#     sem_age_group = []
    
#     for age_group_evkds in evkds_all:
#         mean_subjects = (np.mean(age_group_evkds, axis=0))*1e6
        
#         mean_age_group.append(mean_subjects)
      
#     mean_data[condition] = mean_age_group

# # Replace this with difference calculated from 5-second evoked response
# a = np.mean(mean_data[0][0],axis=0)
 
# array_to_plot_as_topomap = np.zeros((32,len(a)))

# array_to_plot_as_topomap = a[:]



# fig=mne.viz.plot_topomap(array_to_plot_as_topomap, xy, contours=15, res=128, size=4.5)
# # plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.88)
# plt.tight_layout()
# plt.rcParams['axes.titlepad'] = 0
# plt.title('Across Age', fontsize=14,loc='center')

# plt.savefig(save_loc + 'GDT_All_Topomap', dpi=300)