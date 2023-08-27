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

plt.switch_backend('QT5Agg')  # Making the plots interactive (Scrollable)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# Defining the dimensions and quality of figures
plt.rcParams["figure.figsize"] = (5.5,5)
plt.rcParams['figure.dpi'] = 500
plt.rc('xtick', labelsize=11) 
plt.rc('ytick', labelsize=11) 

#%%Setting up stuff
fig_loc = 'C:/Users/vmysorea/Desktop/PhD/GreenLightMeeting/Figures/'
save_mat_loc = 'D:/PhD/Data/Binding_matfiles/1-40Hz/'
data_loc = 'C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/MTB_Analysis/'

subjlist = ['S104']
subjlist = ['S273', 'S069', 'S072', 'S078', 'S088', 
            'S104', 'S260', 'S268', 'S269', 'S270', 
            'S271', 'S272', 'S274', 'S277', 'S279', 
            'S280', 'S281', 'S282', 'S284', 'S285', 
            'S288', 'S290', 'S291', 'S303', 'S305', 
            'S308', 'S309', 'S310', 'S312', 'S337', 
            'S339', 'S340', 'S341', 'S342', 'S344', 
            'S345', 'S347', 'S352', 'S355', 'S358']


# for subj in range(len(subjlist)):
#     sub = subjlist [subj]
#     dat1 = io.loadmat(save_mat_loc + sub + '_1-40Hz_Evoked_AllChan.mat', squeeze_me=True)
#     dat1.keys()
#     t = dat1['t']
#     t_full = dat1['t_full']

#Load data with subjects' age and gender and pick subj only from subjlist 
dat = pd.read_csv(data_loc + 'subj_age_gender.csv')
dat0 = dat['Subject'].isin(subjlist)
dat = dat[dat0]

#Categorizing into different age groups 
def group_age (age):
    if age <= 35:
        return 'YNH'
    elif age <=55:
        return 'MNH'
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

mat_agegroups = []
picks = [4, 25, 30, 31]         #Took off the Cs

for age, groups in age_groups:
    group_evkds0 = []  # Initialize lists for each condition and age group
    group_evkds1 = []
    group_evkds2 = []
    group_evkds3 = []
    group_evkds4 = []
    group_evkds5 = []
    group_evkds6 = []
    group_evkds7 = []
    
    for index, column in groups.iterrows():
        subj = column['Subject']
        dat = io.loadmat(save_mat_loc + subj + '_1-40Hz_Evoked_AllChan.mat', squeeze_me=True)
        dat.keys()
        evkd0 = dat['evkd0'][picks]
        evkd1 = dat['evkd1'][picks]
        evkd2 = dat['evkd2'][picks]
        evkd3 = dat['evkd3'][picks]
        evkd4 = dat['evkd4'][picks]
        evkd5 = dat['evkd5'][picks]
        evkd6 = dat['evkd6'][picks]
        evkd7 = dat['evkd7'][picks]

        group_evkds0.append(evkd0.mean(axis=0))
        group_evkds1.append(evkd1.mean(axis=0))
        group_evkds2.append(evkd2.mean(axis=0))
        group_evkds3.append(evkd3.mean(axis=0))
        group_evkds4.append(evkd4.mean(axis=0))
        group_evkds5.append(evkd5.mean(axis=0))
        group_evkds6.append(evkd6.mean(axis=0))
        group_evkds7.append(evkd7.mean(axis=0))

    # Append data for each age group to lists
    evkds0_all.append(group_evkds0)
    evkds1_all.append(group_evkds1)
    evkds2_all.append(group_evkds2)
    evkds3_all.append(group_evkds3)
    evkds4_all.append(group_evkds4)
    evkds5_all.append(group_evkds5)
    evkds6_all.append(group_evkds6)
    evkds7_all.append(group_evkds7)
   
#%%
conditions = {
    0: evkds0_all,
    1: evkds1_all,
    2: evkds2_all,
    3: evkds3_all,
    4: evkds4_all,
    5: evkds5_all,
}

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

#%% Plot each condition in separate plots 
# Define condition names
condition_names = { 0: '12 Onset',
                    1: '20 Onset',
                    2: '12 Incoherent to Coherent',
                    3: '12 Coherent to Incoherent', 
                    4: '20 Incoherent to Coherent',
                    5: '20 Coherent to Incoherent',
                    }

age_group_labels = {'MNH': 'Middle (36-55 y)',
                    'ONH': 'Old (>=56 y)',
                    'YNH': 'Young (<=35 y)'}

# Iterate through conditions
for condition, evkds_all in conditions.items():
    plt.figure()  # Create a new plot for each condition
    plt.title(condition_names.get(condition, f'Condition {condition}'))
    
    # Iterate through age groups
    for age_group, (mean_age_group, sem_age_group) in enumerate(zip(mean_data[condition], sem_data[condition])):
        age_label = list(age_group_labels.keys())[age_group]  # Get age group label based on index
        age_label_text = age_group_labels.get(age_label, age_label)  # Get label text or fallback to key
        
        # Plot mean with SEM as shaded region
        plt.plot(t, mean_age_group, label=f'{age_label_text} Age Group')
        plt.fill_between(t, mean_age_group - sem_age_group, mean_age_group + sem_age_group, alpha=0.3)
    
    plt.xlabel('Time')
    plt.ylabel('Mean')
    plt.legend()
    plt.grid()
    plt.show()  # Show the plot for this condition

#%% Plot full 5 seconds 
conditions_full = {6: evkds6_all,
              7: evkds7_all}
condition_names_full = { 6: '12 Full 5 seconds',
                    7: '20 Full 5 seconds'}

age_group_labels = {'MNH': 'Middle (36-55 y)',
                    'ONH': 'Old (>=56 y)',
                    'YNH': 'Young (<=35 y)'}

# Iterate through conditions
for condition, evkds_all in conditions_full.items():
    plt.figure()  # Create a new plot for each condition
    plt.title(condition_names_full.get(condition, f'Condition {condition}'))
    
    # Iterate through age groups
    for age_group, (mean_age_group, sem_age_group) in enumerate(zip(mean_data[condition], sem_data[condition])):
        age_label = list(age_group_labels.keys())[age_group]  # Get age group label based on index
        age_label_text = age_group_labels.get(age_label, age_label)  # Get label text or fallback to key
        
        # Plot mean with SEM as shaded region
        plt.plot(t_full, mean_age_group, label=f'{age_label_text} Age Group')
        plt.fill_between(t_full, mean_age_group - sem_age_group, mean_age_group + sem_age_group, alpha=0.3)
    
    plt.xlabel('Time')
    plt.ylabel('Mean')
    plt.legend()
    plt.grid()
    plt.show()  # Show the plot for this condition
    
#%% Subplots!

# Define condition names
condition_names = { 0: '12 Onset',
                    1: '20 Onset',
                    2: '12 Incoherent to Coherent',
                    3: '12 Coherent to Incoherent', 
                    4: '20 Incoherent to Coherent',
                    5: '20 Coherent to Incoherent',
                    }

# Define age group labels
age_group_labels = {'YNH': 'Young (<=35 y)',
                    'MNH': 'Middle (36-55 y)',
                    'ONH': 'Old (>=56 y)'}

cond_groups = [(0,1), (2,4), (3,5)]

# Create a figure with 3 horizontal subplots
for cond in cond_groups:  
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex= True)

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
            ax.legend()
        
        # ax.set_ylabel()
        # ax.set_ylim(-2,5.2)
        ax.set_xlim(-0.1,1.1)
        ax.grid()
       
        fig.text(-0.0001, 0.5, 'Amplitude (\u03bcV)', va='center', rotation='vertical', fontsize=12)
        plt.xlabel('Time (s)', fontsize =12)
        # fig.suptitle(f'{condition_name}', size=16, y=1.001)
    
        plt.tight_layout()
        plt.show()  # Show the plot

plt.savefig(fig_loc + 'Onset12vs20_AcrossAge.png', dpi=500)
