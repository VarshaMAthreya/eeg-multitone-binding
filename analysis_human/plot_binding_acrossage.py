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
import matplotlib.patches as patches

plt.switch_backend('QT5Agg')  # Making the plots interactive (Scrollable)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# Defining the dimensions and quality of figures
plt.rcParams["figure.figsize"] = (5.5,5)
plt.rcParams['figure.dpi'] = 500
# plt.rc('xtick', labelsize=8)
# plt.rc('ytick', labelsize=11)

#%%Setting up stuff
fig_loc = 'D:/PhD/FinalPaper/'
save_mat_loc = 'D:/PhD/Data/Binding_matfiles/0.5-40Hz/'
data_loc = 'D:/PhD/Stim_Analysis/MTB_Analysis/'

# subjlist = ['S104']
subjlist = ['S069', 'S072', 'S078', 'S088', 'S104',
            'S105', 'S207', 'S211', 'S259', 'S260',
            'S268', 'S269', 'S270', 'S271', 'S272',
            'S273', 'S274', 'S277', 'S279', 'S280',
            'S281', 'S282', 'S284', 'S285', 'S288',
            'S290', 'S291', 'S303', 'S305', 'S308',
            'S309', 'S310', 'S312', 'S337', 'S339',
            'S340', 'S341', 'S344', 'S345', 'S352',
            'S355', 'S358']


for subj in range(len(subjlist)):
    sub = subjlist [subj]
    dat1 = io.loadmat(save_mat_loc + sub + '_0.5-40Hz_Evoked_AllChan.mat', squeeze_me=True)
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
evkds_onset12_all = []
evkds_onset20_all = []
evkds_coh12_all = []
evkds_incoh12_all = []
evkds_coh20_all = []
evkds_incoh20_all = []
evkds_full12_all = []
evkds_full20_all = []
# ss_all12 =[]
# ss_all20 =[]
# gfp_all12 = []
# gfp_all20 = []

mat_agegroups = []
picks = [4, 25, 30, 31]
# picks_ss = [3, 30, 26, 4, 25, 7, 31, 22, 8, 21, 11, 12, 18]

for age, groups in age_groups:
    group_evkds_onset12 = []  # Initialize lists for each condition and age group
    group_evkds_onset20 = []
    group_evkds_coh12 = []
    group_evkds_incoh12 = []
    group_evkds_coh20 = []
    group_evkds_incoh20 = []
    group_evkds_full12 = []
    group_evkds_full20 = []
    # group_ss12 =[]
    # group_ss20 =[]
    # group_gfp12 =[]
    # group_gfp20 = []

    for index, column in groups.iterrows():
        subj = column['Subject']
        dat = io.loadmat(save_mat_loc + subj + '_0.5-40Hz_Evoked_AllChan.mat', squeeze_me=True)
        dat.keys()
        evkd_onset12 = dat['evkd_onset12'][picks]
        evkd_onset20 = dat['evkd_onset20'][picks]
        evkd_coh12 = dat['evkd_coh12'][picks]
        evkd_incoh12 = dat['evkd_incoh12'][picks]
        evkd_coh20 = dat['evkd_coh20'][picks]
        evkd_incoh20 = dat['evkd_incoh20'][picks]
        evkd_full12 = dat['evkd_full12'][picks]
        evkd_full20 = dat['evkd_full20'][picks]

        # ss6 = dat['evkd6'][picks_ss]      ###Doing this only for the 5 sec plot
        # ss7 = dat['evkd7'][picks_ss]
        # ss12 = ((ss6**2).sum(axis=0))
        # ss20 = ((ss7**2).sum(axis=0))

        # gfp12 = ss6.std(axis=0)
        # gfp20 = ss7.std(axis=0)

        group_evkds_onset12.append(evkd_onset12.mean(axis=0))
        group_evkds_onset20.append(evkd_onset20.mean(axis=0))
        group_evkds_coh12.append(evkd_coh12.mean(axis=0))
        group_evkds_incoh12.append(evkd_incoh12.mean(axis=0))
        group_evkds_coh20.append(evkd_coh20.mean(axis=0))
        group_evkds_incoh20.append(evkd_incoh20.mean(axis=0))
        group_evkds_full12.append(evkd_full12.mean(axis=0))
        group_evkds_full20.append(evkd_full20.mean(axis=0))

        # group_ss12.append(ss12)
        # group_ss20.append(ss20)

        # group_gfp12.append(gfp12)
        # group_gfp20.append(gfp20)

    # Append data for each age group to lists
    evkds_onset12_all.append(group_evkds_onset12)
    evkds_onset20_all.append(group_evkds_onset20)
    evkds_coh12_all.append(group_evkds_coh12)
    evkds_incoh12_all.append(group_evkds_incoh12)
    evkds_coh20_all.append(group_evkds_coh20)
    evkds_incoh20_all.append(group_evkds_incoh20)
    evkds_full12_all.append(group_evkds_full12)
    evkds_full20_all.append(group_evkds_full20)

    # ss_all12.append(group_ss12)
    # ss_all20.append(group_ss20)

    # gfp_all12.append(group_gfp12)
    # gfp_all20.append(group_gfp20)

#%% Calculating the mean and SEM within each age group for each condition
conditions = {
    0: evkds_onset12_all,
    1: evkds_onset20_all,
    2: evkds_coh12_all,
    3: evkds_incoh12_all,
    4: evkds_coh20_all,
    5: evkds_incoh20_all,
    6: evkds_full12_all,
    7: evkds_full20_all}
    # 8: ss_all12,
    # 9: ss_all20,
    # 10: gfp_all12,
    # 11: gfp_all20}

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

#%% Plot 12 vs 20 for every condition for across age groups

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

cond_groups = [(4,5)]

# cond_groups = [(6,7)]

# Create a figure with 2 horizontal subplots
for cond in cond_groups:
    fig, axs = plt.subplots(2, 1, figsize=(6, 5), sharex= True)

# Iterate through age groups
    for age_group_index, age_group in enumerate(age_group_labels.keys()):

        ax = axs[age_group_index]
        # N = age_groups['Subject'].count()
        ax.set_title(f'Age Group: {age_group_labels[age_group]}', size =6, pad =0)

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
        ax.set_ylim(3, -3)
        # ax.set_xlim(-0.1,1.1)
        ax.grid()

        fig.text(-0.0001, 0.5, 'Amplitude (\u03bcV)', va='center', rotation='vertical', fontsize=10)
        plt.xlabel('Time (s)', fontsize =10)
        # fig.suptitle(f'{condition_name}', size=16, y=1.001)

    # plt.tight_layout()
    # plt.savefig(fig_loc + f'cond_{cond[0]}_{cond[1]}_1.png', dpi = 500)
    # plt.close()
    plt.show()

# plt.savefig(fig_loc + 'Onset12vs20_AcrossAge.png', dpi=500)

#%%## Plot all age groups in same subplot

# Condition and age group labels
condition_names = {
    0: '12 Onset', 1: '20 Onset',
    2: '12 Incoherent to Coherent', 3: '12 Coherent to Incoherent',
    4: '20 Incoherent to Coherent', 5: '20 Coherent to Incoherent',
    6: '12 Tone', 7: '20 Tone',
    8: '12 Tone Sum of Squares', 9: '20 Tone Sum of Squares',
    10: '12 Tone | GFP', 11: '20 | GFP'
}

age_group_labels = {
    'YNH': 'Young (≤40 y)',
    'ONH': 'Old (>41 y)'
}

sns.set_palette("Dark2")

fig, axs = plt.subplots(3, 1, figsize=(6, 5), sharex=True)

for idx, condition in enumerate([1, 4, 5]):
    ax = axs[idx]
    ax.set_title(condition_names[condition], fontsize=8, weight='bold')

    legend_text = []

    for age_idx, age_key in enumerate(age_group_labels.keys()):
        label = age_group_labels[age_key]
        mean = mean_data[condition][age_idx]
        sem = sem_data[condition][age_idx]
        N = age_groups['age_group'].value_counts().get(age_key, 0)

        ax.plot(t, mean, label=f"{label} (N={N})", linewidth=1.5)
        ax.fill_between(t, mean - sem, mean + sem, alpha=0.25)

        ax.tick_params(axis="y", labelsize=8)

    if idx == 0:
        ax.legend(loc='upper right', fontsize=5)

    ax.set_xlim(-0.1, 1.1)
    ax.grid(True, linestyle='--', linewidth=0.5)

# Styling for entire figure
# plt.subplots_adjust(hspace=0.3)
plt.xticks(fontsize=8)
plt.xlabel('Time (s)', fontsize=10)
fig.text(0.01, 0.5, 'Amplitude (μV)', va='center', rotation='vertical', fontsize=10)
plt.tight_layout()
fig.suptitle('Picks - Cz, Fz, FC1, FC2', x=1, ha='right', fontsize=8)

plt.show()

plt.savefig(fig_loc + 'Binding20_AcrossAges.png', dpi = 500, bbox_inches='tight')


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
                     9: '20 Tone | Sum of Squares | Cz, Fz, FC1, FC2',
                     10: '12 Tone | GFP',
                     11: '20 Tone | GFP'}

# Define age group labels
# age_group_labels = {'YNH': 'Young (<36)', 'MNH': 'Middle (36-55)', 'ONH': 'Old (>55)'}

sns.set_palette ("Dark2")

age_group_labels = {'YNH': 'Young (<=40 y)',
                    'ONH': 'Old (>41 y)'}

fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)

condition_to_plot = 9

# Plot the selected condition in the subplot
# ax.set_title(condition_names[condition_to_plot], fontsize = 16)

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

ax.legend(labels=legend_text, loc='upper right', fontsize='small')

ax.set_ylim(0, 3*1e-4)
ax.set_xlim(-0.2, 5.5)
# ax.grid()

for x_value in (0, 1, 2, 3, 4, 5) :
    ax.axvline(x=x_value, color='black', linestyle='--', alpha=1)

y_limits = ax.get_ylim()

# labels = ["Stim On", "Stim Off"]
# for x, label in zip([0,5], labels):
#     ax.text(x, y_limits[1] + 0.1, label, ha='center',weight='bold')

# coh_labels = ["Incoherent", "Coherent", "Incoherent", "Coherent","Incoherent"]
# for x, label in zip([0.55,1.5,2.5,3.5,4.5], coh_labels):
#     ax.text(x, y_limits[1] - 1.1, label, ha='center', weight='bold', fontsize = 14)

# shaded_regions = [(1.35, 1.8), (2.35, 2.8), (3.35, 3.8), (4.35, 4.8)]
# for start, end in shaded_regions:
#     ax.fill_between(x=[start, end], y1=y_limits[0], y2=y_limits[1], color='gray', alpha=0.3)

#For SS
labels = ["Stim On", "Stim Off"]
for x, label in zip([0,5], labels):
    ax.text(x, y_limits[1] + 0.05e-4, label, ha='center',weight='bold')

coh_labels = ["Incoherent", "Coherent", "Incoherent", "Coherent","Incoherent"]
for x, label in zip([0.55,1.4,2.4,3.4,4.4], coh_labels):
    ax.text(x, y_limits[1] - 0.5e-4, label, ha='center', weight='bold')

shaded_regions = [(1.35, 1.8), (2.35, 2.8), (3.35, 3.8), (4.35, 4.8)]
for start, end in shaded_regions:
    ax.fill_between(x=[start, end], y1=y_limits[0], y2=y_limits[1], color='gray', alpha=0.3)

plt.xlabel('Time (s)', fontsize=16)
# plt.ylabel('Amplitude (\u03bcV)', fontsize=16)
plt.ylabel('Sum of Squares (\u03bcV $\mathregular{^{2}}$)', fontsize=16)
# plt.suptitle('20 Tone | Picks - Cz, Fz, FC1, FC2', x=0.8, ha='right', fontsize=16)
plt.suptitle('20 Tone | Picks - Cz, Fz, FC1, FC2', fontsize=16)
plt.tight_layout()
plt.show()

# plt.savefig(fig_loc + "ARO24_SS20_FullTime_1-40Hz_AcrossAge.png", dpi=500)

#%% Plot topomaps for 5 second duration

import mne
from mne.channels.layout import _pol_to_cart, _cart_to_sph

montage = mne.channels.make_standard_montage('biosemi32')

chs = montage._get_ch_pos()
ch_names, xyz = zip(*[(ich, ixyz) for ich, ixyz in chs.items()])
xyz = np.vstack(xyz)

mne.viz.plot_montage(montage)
sph = _cart_to_sph(xyz)
xy = _pol_to_cart(sph[:, 1:][:, ::-1]) * 0.05

### Calculating times for response, steady-state for onset, incoherent-coherent and coherent-incoherent transitions

intervals = [(0, 0.29), (0.3, 0.8), (1.0, 1.29), (1.3, 1.8), (2.0, 2.29), (2.3, 2.8), (3.0, 3.29), (3.3, 3.8), (4.0, 4.29), (4.3,4.8), (5.0,5.29)]

t_values = []
for start, end in intervals:
    t_values.append((t_full >= start) & (t_full <= end))



# conditions = {6: evkds_full12_all,
#               7: evkds_full20_all}

# evoked12 = []
# evoked20 = []
# for a in t_values:
#     for subjlist in conditions.values():
#         evoked_subj12 = [arr[a] for arr in subjlist]
#         evoked12.append(evoked_subj12)
#     evokeds12.append(evoked12)
#     evokeds20.append(evoked20)


#Getting the evoked responses for the above specified times
evokeds12 = []
evokeds20 = []

for i in t_values:
    evoked12 = []
    for subjlist in evkds_full12_all:
        evoked_subj = []
        for arr in subjlist:
            evoked_subj.append(arr[i])
        evoked12.append(evoked_subj)
    evokeds12.append(evoked12)

for i in t_values:
    evoked20 = []
    for subjlist in evkds_full20_all:
        evoked_subj = []
        for arr in subjlist:
            evoked_subj.append(arr[i])
        evoked20.append(evoked_subj)
    evokeds20.append(evoked20)

#%% Calculating the mean and SEM within each age group for each condition
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
# array_to_plot_as_topomap = np.mean(mean_data,axis=0)


# # Replace this with difference calculated from 5-second evoked response
# array_to_plot_as_topomap = evoked.mean(axis=1)

# fig=mne.viz.plot_topomap(array_to_plot_as_topomap, xy, contours=15, res=128, size=4.5)
# # plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.88)
# plt.tight_layout()
# plt.rcParams['axes.titlepad'] = 0
# plt.title('Across Age', fontsize=14,loc='center')

# plt.savefig(save_loc + 'GDT_All_Topomap', dpi=300)