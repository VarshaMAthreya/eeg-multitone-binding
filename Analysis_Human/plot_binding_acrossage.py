# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 17:54:46 2023

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

plt.switch_backend('QT5Agg')  # Making the plots interactive (Scrollable)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# Defining the dimensions and quality of figures
plt.rcParams["figure.figsize"] = (5.5,5)
plt.rcParams['figure.dpi'] = 500
plt.rc('xtick', labelsize=11) 
plt.rc('ytick', labelsize=11) 

#%%Setting up stuff
fig_loc = 'C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/MTB_Analysis/GreenLight/'
save_mat_loc = 'D:/PhD/Data/Binding_matfiles/1-40Hz/'
data_loc = 'C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/MTB_Analysis/'

# subjlist = ['S104']
subjlist = ['S273', 'S069', 'S072', 'S078', 'S088', 
            'S105', 'S207', 'S259', 'S260', 'S268', 
            'S269', 'S270', 'S271', 'S272', 'S274', 
            'S277', 'S279', 
            'S280', 'S281', 'S282', 'S284', 'S285', 
            'S288', 'S290', 'S291', 'S303', 'S305', 
            'S308', 'S309', 'S310', 'S312', 'S337', 
            'S339', 'S340', 'S341', 'S342', 'S344', 
            'S345', 'S347', 'S352', 'S355', 'S358']


for subj in range(len(subjlist)):
    sub = subjlist [subj]
    dat1 = io.loadmat(save_mat_loc + sub + '_0.4-40Hz_Evoked_AllChan.mat', squeeze_me=True)
    dat1.keys()
    t = dat1['t']
    t_full = dat1['t_full']

#Load data with subjects' age and gender and pick subj only from subjlist 
dat = pd.read_csv(data_loc + 'subj_age_gender.csv')
dat0 = dat['Subject'].isin(subjlist)
dat = dat[dat0]

#Categorizing into different age groups 
# def group_age (age):
#     if age <= 35:
#         return 'YNH'
#     elif age <=55:
#         return 'MNH'
#     else: 
#         return 'ONH'
    
def group_age(age):
    if age <= 40:
        return 'YNH'
    else:
        return 'ONH'

    
dat['age_group'] = dat['Age'].apply([group_age])
age_groups = dat.groupby(['age_group'], sort=False)

#%% Loading all subjects' data 

# 0 - 12 Onset (Upto 1.1 s) -- 0
# 1 - 20 Onset (Upto 1.1 s) -- 1
# 10 - 12 Incoherent to Coherent -- 2
# 11 - 12 Coherent to Incoherent -- 3
# 12 - 20 Incoherent to Coherent -- 4
# 13 - 20 Coherent to Incoherent -- 5
# 14 - 12 Full 5 seconds -- 6
# 15 - 20 Full 5 seconds -- 7 

# Initialize empty lists to store data for different conditions and age groups
evkds0_all = []
evkds1_all = []
evkds2_all = []
evkds3_all = []
evkds4_all = []
evkds5_all = []
evkds6_all = []
evkds7_all = []
ss_all12 =[]
ss_all20 =[]
gfp_all12 = []
gfp_all20 = []

mat_agegroups = []
picks = [4, 25, 30, 31]         #Took off the Cs

picks_ss = [3, 30, 26, 4, 25, 7, 31, 22, 8, 21, 11, 12, 18]


for age, groups in age_groups:
    group_evkds0 = []  # Initialize lists for each condition and age group
    group_evkds1 = []
    group_evkds2 = []
    group_evkds3 = []
    group_evkds4 = []
    group_evkds5 = []
    group_evkds6 = []
    group_evkds7 = []
    group_ss12 =[]
    group_ss20 =[]
    group_gfp12 =[]
    group_gfp20 = []
    
    for index, column in groups.iterrows():
        subj = column['Subject']
        dat = io.loadmat(save_mat_loc + subj + '_0.4-40Hz_Evoked_AllChan.mat', squeeze_me=True)
        dat.keys()
        evkd0 = dat['evkd0'][picks]
        evkd1 = dat['evkd1'][picks]
        evkd2 = dat['evkd2'][picks]
        evkd3 = dat['evkd3'][picks]
        evkd4 = dat['evkd4'][picks]
        evkd5 = dat['evkd5'][picks]
        evkd6 = dat['evkd6'][picks]
        evkd7 = dat['evkd7'][picks]
        
        ss6 = dat['evkd6'][picks_ss]      ###Doing this only for the 5 sec plot 
        ss7 = dat['evkd7'][picks_ss]
        ss12 = ((ss6**2).sum(axis=0))
        ss20 = ((ss7**2).sum(axis=0))
        
        gfp12 = ss6.std(axis=0)
        gfp20 = ss7.std(axis=0)

        group_evkds0.append(evkd0.mean(axis=0))
        group_evkds1.append(evkd1.mean(axis=0))
        group_evkds2.append(evkd2.mean(axis=0))
        group_evkds3.append(evkd3.mean(axis=0))
        group_evkds4.append(evkd4.mean(axis=0))
        group_evkds5.append(evkd5.mean(axis=0))
        group_evkds6.append(evkd6.mean(axis=0))
        group_evkds7.append(evkd7.mean(axis=0))
        
        group_ss12.append(ss12)
        group_ss20.append(ss20)
        
        group_gfp12.append(gfp12)
        group_gfp20.append(gfp20)
        
    # Append data for each age group to lists
    evkds0_all.append(group_evkds0)
    evkds1_all.append(group_evkds1)
    evkds2_all.append(group_evkds2)
    evkds3_all.append(group_evkds3)
    evkds4_all.append(group_evkds4)
    evkds5_all.append(group_evkds5)
    evkds6_all.append(group_evkds6)
    evkds7_all.append(group_evkds7)
    
    ss_all12.append(group_ss12)
    ss_all20.append(group_ss20)
    
    gfp_all12.append(group_gfp12)
    gfp_all20.append(group_gfp20)

#%%
conditions = {
    0: evkds0_all,
    1: evkds1_all,
    2: evkds2_all,
    3: evkds3_all,
    4: evkds4_all,
    5: evkds5_all,
    6: evkds6_all,
    7: evkds7_all, 
    8: ss_all12,
    9: ss_all20, 
    10: gfp_all12,
    11: gfp_all20}

mean_data = {}
sem_data = {}

for condition, evkds_all in conditions.items():
    mean_age_group = []
    sem_age_group = []
    
    for age_group_evkds in evkds_all:
        mean_subjects = (np.mean(age_group_evkds, axis=0))*1e6
        sem_subjects = (sem(age_group_evkds, axis=0))*1e6
        
        mean_age_group.append(mean_subjects)
        sem_age_group.append(sem_subjects)
    
    mean_data[condition] = mean_age_group
    sem_data[condition] = sem_age_group
    
#%% Subplots!

###Plotting 12 vs 20 for every condition for across age groups 

# Define condition names
condition_names = { 0: '12 Onset',
                     1: '20 Onset',
                     2: '12 Incoherent to Coherent',
                     3: '12 Coherent to Incoherent', 
                     4: '20 Incoherent to Coherent',
                     5: '20 Coherent to Incoherent',
                     6: '12 Tone', 7: '20 Tone', 
                     8: '12 Tone Sum of Squares', 
                     9: '20 Tone Sum of Squares',
                     10: '12 Tone | GFP', 
                     11: '20 | GFP'}
                                   
                    
# Define age group labels
# age_group_labels = {'YNH': 'Young (<=35 y)',
#                     'MNH': 'Middle (36-55 y)',
#                     'ONH': 'Old (>=56 y)'}

age_group_labels = {'YNH': 'Young (<=40 y)',
                    'ONH': 'Old (>41 y)'}

cond_groups = [(6,7)]

# cond_groups = [(6,7)]

# Create a figure with 3 horizontal subplots
for cond in cond_groups:  
    fig, axs = plt.subplots(3, 1, figsize=(4.5,4), sharex= True)

# Iterate through age groups
    for age_group_index, age_group in enumerate(age_group_labels.keys()):
        
        ax = axs[age_group_index]
        # N = age_groups['Subject'].count()
        ax.set_title(f'Age Group: {age_group_labels[age_group]}', size =10, pad =0)
        
        # Iterate through conditions

        for condition in cond:
            mean_age_group = mean_data[condition][age_group_index]
            sem_age_group = sem_data[condition][age_group_index]
            
            condition_name = condition_names.get(condition, f'Condition {condition}')
            
            # Plot mean with SEM as shaded region
            ax.plot(t, mean_age_group, label=f'{condition_name}')
            ax.fill_between(t, mean_age_group - sem_age_group, mean_age_group + sem_age_group, alpha=0.3)
                
        if age_group_index == 0:
            ax.legend(loc ='upper right',fontsize = 'xx-small' )
        
        # ax.set_ylabel()
        # ax.set_ylim(-2,5.2)
        ax.set_xlim(-0.1,1.1)
        ax.grid()
       
        fig.text(-0.0001, 0.5, 'Amplitude (\u03bcV)', va='center', rotation='vertical', fontsize=12)
        plt.xlabel('Time (s)', fontsize =12)
        # fig.suptitle(f'{condition_name}', size=16, y=1.001)
    
    plt.tight_layout()
    # plt.savefig(fig_loc + f'cond_{cond[0]}_{cond[1]}_1.png', dpi = 500)
    # plt.close()
    plt.show()  

# plt.savefig(fig_loc + 'Onset12vs20_AcrossAge.png', dpi=500)

#%%## all three age groups in same subplot

condition_names = { 0: '12 Onset',
                     1: '20 Onset',
                     2: '12 Incoherent to Coherent',
                     3: '12 Coherent to Incoherent', 
                     4: '20 Incoherent to Coherent',
                     5: '20 Coherent to Incoherent',
                     6: '12 Tone', 
                     7: '20 Tone',
                     8: '12 Tone Sum of Squares', 
                     9: '20 Tone Sum of Squares',
                     10: '12 Tone | GFP', 
                     11: '20 | GFP'}

# Define age group labels
# age_group_labels = {'YNH': 'Young (<36)', 'MNH': 'Middle (36-55)', 'ONH': 'Old (>55)'}

sns.set_palette ("Dark2")

age_group_labels = {'YNH': 'Young (<=40 y)',
                    'ONH': 'Old (>41 y)'}

# Create a figure with 3 subplots

fig, axs = plt.subplots(2, 1, figsize=(6.5, 5), sharex=True, constrained_layout=True)

# Loop through conditions and plot in subplots
for condition_index, condition in enumerate([6, 7]):
    ax = axs[condition_index]
    ax.set_title(condition_names[condition])
    
    legend_text=[]
    
    # Iterate through age groups
    for age_group_index, age_group in enumerate(age_group_labels.keys()):
        mean_age_group = mean_data[condition][age_group_index]
        sem_age_group = sem_data[condition][age_group_index]
        
        N = age_groups['age_group'].count()[age_group]
        
        # Plot mean with SEM as shaded region
        ax.plot(t_full, mean_age_group, label=age_group, alpha=0.9)
        ax.fill_between(t_full, mean_age_group - sem_age_group, mean_age_group + sem_age_group, alpha=0.3)
        
        legend_text.append(f"{age_group} (N={N})")
        
    if condition_index == 0:
        ax.legend(labels=legend_text, loc='upper right', fontsize='xx-small')
    
    # ax.set_ylim(-2,5.1)
    # ax.set_xlim(-0.1,1.1)
    ax.grid()
    
plt.subplots_adjust(wspace=0.15,hspace =0.15)    
plt.xlabel('Time (s)', fontsize =12)
fig.text(0, 0.5, 'Amplitude (\u03bcV)', va='center', rotation='vertical', fontsize=12)
fig.suptitle('Picks - Cz, Fz, FC1, FC2', x=1, ha='right', fontsize=10)
plt.tight_layout()
plt.show()

# plt.savefig(fig_loc + 'Binding20_AcrossAges.png', dpi = 500)


#%% Plotting each condition separately -- Helpful for full time viewing 

condition_names = { 0: '12 Onset',
                     1: '20 Onset',
                     2: '12 Incoherent to Coherent',
                     3: '12 Coherent to Incoherent', 
                     4: '20 Incoherent to Coherent',
                     5: '20 Coherent to Incoherent',
                     6: '12 Tone', 
                     7: '20 Tone | Picks - Cz, Fz, FC1, FC2', 
                     8: '12 Tone | Sum of Squares', 
                     9: '20 Tone | Sum of Squares',
                     10: '12 Tone | GFP', 
                     11: '20 Tone | GFP'}

# Define age group labels
# age_group_labels = {'YNH': 'Young (<36)', 'MNH': 'Middle (36-55)', 'ONH': 'Old (>55)'}

sns.set_palette ("Dark2")

age_group_labels = {'YNH': 'Young (<=40 y)',
                    'ONH': 'Old (>41 y)'}

fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)

condition_to_plot = 9

# Plot the selected condition in the subplot
ax.set_title(condition_names[condition_to_plot])

legend_text = []

# Iterate through age groups
for age_group_index, age_group in enumerate(age_group_labels.keys()):
    mean_age_group = mean_data[condition_to_plot][age_group_index]
    sem_age_group = sem_data[condition_to_plot][age_group_index]

    N = age_groups['age_group'].count()[age_group]

    # Plot mean with SEM as shaded region
    ax.plot(t_full, mean_age_group, label=age_group, alpha=0.9)
    ax.fill_between(t_full, mean_age_group - sem_age_group, mean_age_group + sem_age_group, alpha=0.3)

    legend_text.append(f"{age_group} (N={N})")

ax.legend(labels=legend_text, loc='upper right', fontsize='xx-small')

ax.set_ylim(0, 3.1e-4)
ax.set_xlim(-0.2, 5.5)
# ax.grid()

for x_value in (0, 1, 2, 3, 4, 5) :
    ax.axvline(x=x_value, color='black', linestyle='--', alpha=1)
    
y_limits = ax.get_ylim()

labels = ["Stim On", "Stim Off"]
for x, label in zip([0,5], labels):
    ax.text(x, y_limits[1] + 0.05e-4, label, ha='center',weight='bold')
    
coh_labels = ["Incoherent", "Coherent", "Incoherent", "Coherent","Incoherent"]
for x, label in zip([0.55,1.4,2.4,3.4,4.4], coh_labels):
    ax.text(x, y_limits[1] - 0.5e-4, label, ha='center', weight='bold')
        
shaded_regions = [(1.35, 1.8), (2.35, 2.8), (3.35, 3.8), (4.35, 4.8)]
for start, end in shaded_regions:
    ax.fill_between(x=[start, end], y1=y_limits[0], y2=y_limits[1], color='gray', alpha=0.3)

plt.xlabel('Time (s)', fontsize=12)
# plt.ylabel('Global Field Power (\u03bcV)', fontsize=12)
plt.ylabel('Sum of Squares (\u03bcV $\mathregular{^{2}}$)', fontsize=12)
# plt.suptitle('Picks - Cz, Fz, FC1, FC2', x=0.8, ha='right', fontsize=10)
plt.tight_layout()
plt.show()

plt.savefig(fig_loc + "SS20_FullTime_0.1-40Hz_AcrossAge.png", dpi=500, bbox_inches="tight")