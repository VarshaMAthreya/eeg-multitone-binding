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
#%%Setting up stuff
# save_fig = 'D:/PhD/FinalPaper/'
# save_mat_loc = 'D:/PhD/Data/Binding_matfiles/0.5-40Hz/'

save_fig = 'C:/Users/varsh/Desktop/PhD/FinalPaper/'
save_mat_loc = 'C:/Users/varsh/Desktop/PhD/Data/Binding_matfiles/0.5-40Hz/'

# subjlist = ['S273', 'S069', 'S072', 'S078', 'S088',
#             'S105', 'S207', 'S259', 'S260', 'S268',
#             'S269', 'S270', 'S271', 'S272', 'S274',
#             'S277', 'S279',
#             'S280', 'S281', 'S282', 'S284', 'S285',
#             'S288', 'S290', 'S291', 'S303', 'S305',
#             'S308', 'S309', 'S310', 'S312', 'S337',
#             'S339', 'S340', 'S341', 'S342', 'S344',
#             'S345', 'S347', 'S352', 'S355', 'S358']

subjlist = ['S069', 'S072', 'S078', 'S088', 'S104',
            'S105', 'S207', 'S211', 'S259', 'S260',
            'S268', 'S269', 'S270', 'S271', 'S272',
            'S273', 'S274', 'S277', 'S279', 'S280',
            'S281', 'S282', 'S284', 'S285', 'S288',
            'S290', 'S291', 'S303', 'S305', 'S308',
            'S309', 'S310', 'S312', 'S337', 'S339',
            'S340', 'S341', 'S344', 'S345', 'S352',
            'S355', 'S358']

#Removed - 'S342' - Very bad responses

#%% EVOKED | Loading all subjects' data

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
    dat = io.loadmat(save_mat_loc + sub + '_0.5-40Hz_Evoked_AllChan.mat', squeeze_me=True)
    dat.keys()
    t = dat['t']
    t_full = dat ['t_full']

evkds_onset12 = np.zeros((len(subjlist),t.size))
evkds_onset20 = np.zeros((len(subjlist),t.size))
evkds_coh12 = np.zeros((len(subjlist),t.size))
evkds_incoh12 = np.zeros((len(subjlist),t.size))
evkds_coh20 = np.zeros((len(subjlist),t.size))
evkds_incoh20 = np.zeros((len(subjlist),t.size))
evkds_full12 = np.zeros((len(subjlist),t_full.size))
evkds_full20 = np.zeros((len(subjlist),t_full.size))

# picks = [4, 25, 30, 31]
picks = [31]

for subj in range(len(subjlist)):
    sub = subjlist [subj]
    dat = io.loadmat(save_mat_loc + sub + '_0.5-40Hz_Evoked_AllChan.mat', squeeze_me=True)
    dat.keys()
    evkd_onset12 = dat['evkd_onset12'][picks]
    evkd_onset20 = dat['evkd_onset20'][picks]
    evkd_coh12 = dat['evkd_coh12'][picks]
    evkd_incoh12 = dat['evkd_incoh12'][picks]
    evkd_coh20 = dat['evkd_coh20'][picks]
    evkd_incoh20 = dat['evkd_incoh20'][picks]
    evkd_full12 = dat['evkd_full12'][picks]
    evkd_full20 = dat['evkd_full20'][picks]
    evkds_onset12[subj,:] = (evkd_onset12.mean(axis=0))*1e6      #Taking the mean of 4 fronto-central channels
    evkds_onset20[subj,:] = (evkd_onset20.mean(axis=0))*1e6
    evkds_coh12[subj,:] = (evkd_coh12.mean(axis=0))*1e6
    evkds_incoh12[subj,:] = (evkd_incoh12.mean(axis=0))*1e6
    evkds_coh20[subj,:] = (evkd_coh20.mean(axis=0))*1e6
    evkds_incoh20[subj,:] = (evkd_incoh20.mean(axis=0))*1e6
    evkds_full12[subj,:] = (evkd_full12.mean(axis=0))*1e6
    evkds_full20[subj,:] = (evkd_full20.mean(axis=0))*1e6

#%% EVOKED | Calculating P1, P2 peaks and latencies

### If incoherent to coherent period, then save information from only one peak (vs P1 and P2), as there is usually one discernable peak
# for this period. For onset and coherent-incoherent periods, save both P1 and P2 peaks' information.

# tP1_start, tP1_end = 0, 0.15
# tP2_start, tP2_end = 0.16, 0.3
# t_avg_start, t_avg_end = 0.3, 0.8

tP1_start, tP1_end = 0, 0.1
tP2_start, tP2_end = 0.1, 0.3
t_avg_start, t_avg_end = 0.3, 0.8

binding_results = {}

evkds_all = {'Onset12':evkds_onset12,
              'Onset20':evkds_onset20,
              'Incoherent2Coherent_12':evkds_coh12,
              'Incoherent2Coherent_20':evkds_coh20,
              'Coherent2Incoherent_12':evkds_incoh12,
              'Coherent2Incoherent_20':evkds_incoh20}

# Loop through conditions
for condition, evkds in evkds_all.items():

    # Initialize empty lists for storing values
    if "Incoherent2Coherent" in condition:
        P_peak = []
        P_latency = []
    else:
        P1_peak = []
        P1_latency = []
        P2_peak = []
        P2_latency = []

    avg_steadystate = []

    # Iterate through each subject's evoked dataset
    for subj, evkd in zip(subjlist, evkds):

        # Case 1: Handle Incoherent2Coherent periods (0 to 0.3s)
        if "Incoherent2Coherent" in condition:
            t_p_indices = np.where((t >= 0) & (t <= 0.3))[0]
            if len(t_p_indices) == 0:
                print(f"Warning: Empty time window for P_peak in subject {subj}. Skipping.")
                P_peak.append(np.nan)
                P_latency.append(np.nan)
                continue

            data_p = evkd[t_p_indices]
            peak_index_p = t_p_indices[np.abs(data_p).argmax()]
            peak_latency_p = t[peak_index_p]
            peak_value_p = data_p[np.abs(data_p).argmax()]

            P_peak.append(peak_value_p)
            P_latency.append(peak_latency_p)

        # Case 2: Handle Coherent2Incoherent periods with updated time windows
        elif "Coherent2Incoherent" in condition:
            t1_indices = np.where((t >= 0) & (t <= 0.15))[0]
            t2_indices = np.where((t >= 0.16) & (t <= 0.3))[0]

            # Process P1 (0 to 0.15s)
            if len(t1_indices) == 0:
                print(f"Warning: Empty time window for P1 in subject {subj}. Assigning NaN.")
                P1_peak.append(np.nan)
                P1_latency.append(np.nan)
            else:
                data_t1 = evkd[t1_indices]
                positive_t1_indices = np.where(data_t1 > 0)[0]
                if len(positive_t1_indices) > 0:
                    peak_index_t1 = t1_indices[positive_t1_indices[data_t1[positive_t1_indices].argmax()]]
                    peak_latency_t1 = t[peak_index_t1]
                    peak_value_t1 = data_t1[positive_t1_indices].max()
                else:
                    print(f"Warning: No positive peak for P1 in subject {subj}. Assigning NaN.")
                    peak_value_t1 = np.nan
                    peak_latency_t1 = np.nan

                P1_peak.append(peak_value_t1)
                P1_latency.append(peak_latency_t1)

            # Process P2 (0.16 to 0.3s)
            if len(t2_indices) == 0:
                print(f"Warning: Empty time window for P2 in subject {subj}. Assigning NaN.")
                P2_peak.append(np.nan)
                P2_latency.append(np.nan)
            else:
                data_t2 = evkd[t2_indices]
                positive_t2_indices = np.where(data_t2 > 0)[0]
                if len(positive_t2_indices) > 0:
                    peak_index_t2 = t2_indices[positive_t2_indices[data_t2[positive_t2_indices].argmax()]]
                    peak_latency_t2 = t[peak_index_t2]
                    peak_value_t2 = data_t2[positive_t2_indices].max()
                else:
                    print(f"Warning: No positive peak for P2 in subject {subj}. Assigning NaN.")
                    peak_value_t2 = np.nan
                    peak_latency_t2 = np.nan

                P2_peak.append(peak_value_t2)
                P2_latency.append(peak_latency_t2)

        else:  # Handle Onset periods using original time windows
            t1_indices = np.where((t >= tP1_start) & (t <= tP1_end))[0]
            t2_indices = np.where((t >= tP2_start) & (t <= tP2_end))[0]
            t_avg_indices = np.where((t >= t_avg_start) & (t <= t_avg_end))[0]

            # Handle P1
            if len(t1_indices) == 0:
                print(f"Warning: Empty time window for P1 in subject {subj}. Assigning NaN.")
                P1_peak.append(np.nan)
                P1_latency.append(np.nan)
            else:
                data_t1 = evkd[t1_indices]
                positive_t1_indices = np.where(data_t1 > 0)[0]
                if len(positive_t1_indices) > 0:
                    peak_index_t1 = t1_indices[positive_t1_indices[data_t1[positive_t1_indices].argmax()]]
                    peak_latency_t1 = t[peak_index_t1]
                    peak_value_t1 = data_t1[positive_t1_indices].max()
                else:
                    print(f"Warning: No positive peak for P1 in subject {subj}. Assigning NaN.")
                    peak_value_t1 = np.nan
                    peak_latency_t1 = np.nan

                P1_peak.append(peak_value_t1)
                P1_latency.append(peak_latency_t1)

            # Handle P2
            if len(t2_indices) == 0:
                print(f"Warning: Empty time window for P2 in subject {subj}. Assigning NaN.")
                P2_peak.append(np.nan)
                P2_latency.append(np.nan)
            else:
                data_t2 = evkd[t2_indices]
                positive_t2_indices = np.where(data_t2 > 0)[0]
                if len(positive_t2_indices) > 0:
                    peak_index_t2 = t2_indices[positive_t2_indices[data_t2[positive_t2_indices].argmax()]]
                    peak_latency_t2 = t[peak_index_t2]
                    peak_value_t2 = data_t2[positive_t2_indices].max()
                else:
                    print(f"Warning: No positive peak for P2 in subject {subj}. Assigning NaN.")
                    peak_value_t2 = np.nan
                    peak_latency_t2 = np.nan

                P2_peak.append(peak_value_t2)
                P2_latency.append(peak_latency_t2)

        # Calculate average steady-state response
        avg_response = np.mean(evkd[np.where((t >= t_avg_start) & (t <= t_avg_end))[0]])
        avg_steadystate.append(avg_response)

    # Store results based on condition type
    if "incoherent2coherent" in condition.lower():
        binding_results[condition] = {
            'subject': np.array(subjlist, dtype=object),
            'P_peak': np.array(P_peak, dtype=np.float64).reshape(-1, 1),
            'P_latency': np.array(P_latency, dtype=np.float64).reshape(-1, 1),
            'Avg_steadystate': np.array(avg_steadystate, dtype=np.float64).reshape(-1, 1)
        }
    else:
        binding_results[condition] = {
            'subject': np.array(subjlist, dtype=object),
            'P1_peak': np.array(P1_peak, dtype=np.float64).reshape(-1, 1),
            'P1_latency': np.array(P1_latency, dtype=np.float64).reshape(-1, 1),
            'P2_peak': np.array(P2_peak, dtype=np.float64).reshape(-1, 1),
            'P2_latency': np.array(P2_latency, dtype=np.float64).reshape(-1, 1),
            'Avg_steadystate': np.array(avg_steadystate, dtype=np.float64).reshape(-1, 1)
        }

# Save results to MATLAB file
savemat(save_mat_loc + 'AllSubj_BindingA32_0.5-40Hz_Peaks_Latencies.mat', {'binding_results': binding_results})

#%% EVOKED | Plot evoked responses - Avg of 4 chan
fig, ax = plt.subplots(3, 1, figsize = (6,5), sharex = True)
ax[0].errorbar(t, evkds_onset12.mean(axis=0), yerr=sem(evkds_onset12), label = '12 tone coherence (N='+ str(len(subjlist)) +')', color='green', linewidth=2, ecolor='darkseagreen')
ax[0].errorbar(t, evkds_onset20.mean(axis=0), yerr=sem(evkds_onset20), label = '20 tone coherence (N='+ str(len(subjlist)) +')', color='purple', linewidth=2, ecolor='thistle')
ax[1].errorbar(t, evkds_coh12.mean(axis=0), yerr=sem(evkds_coh12), color='green', linewidth=2, ecolor='darkseagreen')
ax[1].errorbar(t, evkds_coh20.mean(axis=0), yerr=sem(evkds_coh20), color='purple', linewidth=2, ecolor='thistle')
ax[2].errorbar(t, evkds_incoh12.mean(axis=0), yerr=sem(evkds_incoh12), color='green', linewidth=2, ecolor='darkseagreen')
ax[2].errorbar(t, evkds_incoh20.mean(axis=0), yerr=sem(evkds_incoh20), color='purple', linewidth=2, ecolor='thistle')

plt.suptitle('EEG Across-Channel Measure | Picks - Cz, Fz, FC1, FC2 (N='+ str(len(subjlist)) +')')
fig.subplots_adjust(top=0.88)
ax[0].set_title('Onset')
ax[1].title.set_text('Incoherent to Coherent')
ax[2].title.set_text('Coherent to Incoherent')
ax[0].legend()
fig.text(-0.00001, 0.5,'Amplitude (\u03bcV)',fontsize=14, va='center', rotation='vertical', weight = "bold")
# plt.xlim(-0.2, 1.1)
# plt.ylabel('Amplitude (\u03bcV)',fontsize=14)
plt.xlabel('Time(s)',fontsize=14, weight = "bold")
plt.xticks(np.arange(-0.2,1.1,0.2),fontsize = 14)

for i in range(3):
  # ax[i].set_ylim(-2,3)
  ax[i].tick_params(axis='y', which='major', labelsize=14, width=1.5, length=6)

fig.tight_layout()
plt.show()

plt.savefig(save_fig + "0.5-40Hz_AmpAcrossCond(N=42).png", dpi=500, bbox_inches="tight")

#%% EVOKED | Plot full time

fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
plt.errorbar(t_full, evkds_full12.mean(axis=0), yerr=sem(evkds_full12), label = '12 Tone', color='green', linewidth=2, ecolor='darkseagreen')
plt.errorbar(t_full, evkds_full20.mean(axis=0), yerr=sem(evkds_full20), label = '20 Tone', color='purple', linewidth=2, ecolor='thistle')
plt.title('EEG Across-Channel Measure | Picks - Cz, Fz, FC1, FC2 (N='+ str(len(subjlist)) +')')
plt.xlim(-0.2, 5.5)
plt.ylim(-1,5)
ymax = 5
plt.vlines(x=[0,1,2,3,4,5], ymin=-1, ymax= ymax+0.5, colors='black', ls='--',alpha=1)
ax.text(0, ymax+0.1, 'Stim On', va='center', ha='center',  weight='bold')
ax.text(0.5, ymax-0.5, 'Incoherent', va='center', ha='center',  weight='bold')
ax.text(1.5, ymax-0.5, 'Coherent', va='center', ha='center',  weight='bold')
ax.text(2.5, ymax-0.5, 'Incoherent', va='center', ha='center', weight='bold')
ax.text(3.5, ymax-0.5, 'Coherent', va='center', ha='center',  weight='bold')
ax.text(4.5, ymax-0.5, 'Incoherent', va='center', ha='center',  weight='bold')
ax.text(5, ymax+0.1, 'Stim End', va='center', ha='center',  weight='bold')
ax.axvspan(1.3,1.8, alpha=0.3,color='gray')
ax.axvspan(3.3,3.8, alpha=0.3,color='gray')
ax.axvspan(2.3,2.8, alpha=0.3,color='gray')
ax.axvspan(4.3,4.8, alpha=0.3,color='gray')
plt.legend(loc='upper right', fontsize='xx-small')
# fig.tight_layout()
plt.ylabel('Amplitude (\u03bcV)',fontsize=14, weight = "bold")
plt.xlabel('Time(s)',fontsize=14, weight = "bold")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# plt.savefig(save_fig + "FullTime_0.5-40Hz_AmpAcrossCond(N=42).png", dpi=500, bbox_inches="tight")


#%% EVOKED | Calculating DC shift from 300-800 ms (Amplitude) -- Baselined -- 1 second

t1 = t>=0.3
t2 = t<=0.8
t3 = np.array([t2[i] and t1[i] for i in range(len(t1))])

evoked12_coh = evkds_coh12[:,t3].mean(axis=1)
evoked12_incoh = evkds_incoh12[:,t3].mean(axis=1)
evoked20_coh = evkds_coh20[:,t3].mean(axis=1)
evoked20_incoh = evkds_incoh12[:,t3].mean(axis=1)

evoked12 = (evoked12_coh - evoked12_incoh)     #Calculating coherent - incoherent (kinda baselining)
evoked20 = (evoked20_coh - evoked20_incoh)

plt.bar(('12','20'), (evoked12.mean(), evoked20.mean()))
plt.show()

mat_id = dict(sub=subjlist,evoked12 = evoked12, evoked20=evoked20,evoked12_coh=evoked12_coh,evoked12_incoh=evoked12_incoh,
              evoked20_coh=evoked20_coh,evoked20_incoh=evoked20_incoh)

# savemat(save_mat_loc + 'AllSubj_EvokedDiff(4chan)_1-40Hz_1sec(N=39).mat', mat_id)

#%% EVOKED | Plot full time - Without baseline

plt.errorbar(t_full, evkds_full12.mean(axis=0), yerr=sem(evkds_full12), label = '12 tone coherence (N='+ str(len(subjlist)) +')', color='green', linewidth=2, ecolor='darkseagreen')
plt.errorbar(t_full, evkds_full20.mean(axis=0), yerr=sem(evkds_full20), label = '20 tone coherence (N='+ str(len(subjlist)) +')', color='purple', linewidth=2, ecolor='thistle')
plt.suptitle('Binding (A32) - Amplitude (N='+ str(len(subjlist)) +')')
plt.subplots_adjust(top=0.88)
plt.legend()
# fig.tight_layout()
plt.ylabel('Amplitude (\u03bcV)')
plt.xlabel('Time(s)',fontsize=14)
plt.rcParams["figure.figsize"] = (6.5,5)
plt.xticks(fontsize=14)
plt.show()

#%% EVOKED | Calculating DC shift from 300-800 ms - 5 seconds duration
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

evoked12_1=evkds_full12[:,t3]
evoked12_2=evkds_full12[:,t6]
evoked12_3=evkds_full12[:,t9]
evoked12_4=evkds_full12[:,t12]
evoked12_5=evkds_full12[:,t15]

evoked20_1=evkds_full20[:,t3]
evoked20_2=evkds_full20[:,t6]
evoked20_3=evkds_full20[:,t9]
evoked20_4=evkds_full20[:,t12]
evoked20_5=evkds_full20[:,t15]

evoked12_coh = evoked12_2 + evoked12_4
evoked12_incoh =  evoked12_3 + evoked12_5
evokeds12_5sec=(evoked12_coh-evoked12_incoh).mean(axis=1)

evoked20_coh = evoked20_2 + evoked20_4
evoked20_incoh =  evoked20_3 + evoked20_5
evokeds20_5sec=(evoked20_coh-evoked20_incoh).mean(axis=1)

plt.bar(('12','20'), (evokeds12_5sec.mean(), evokeds20_5sec.mean()))
plt.show()

mat_ids1=dict(subj=subjlist,evokeds12_5sec=evokeds12_5sec, evokeds20_5sec=evokeds20_5sec)

# savemat(save_mat_loc + 'AllSubj_A32_EvokedDiff_0.4-40Hz_5sec(N=39).mat', mat_ids1)

#%% GFP and SS

# gfps0 = np.zeros((len(subjlist),t.size))
# gfps1 = np.zeros((len(subjlist),t.size))
# gfps2 = np.zeros((len(subjlist),t.size))
# gfps3 = np.zeros((len(subjlist),t.size))
# gfps4 = np.zeros((len(subjlist),t.size))
# gfps5 = np.zeros((len(subjlist),t.size))
# gfps6 = np.zeros((len(subjlist),t_full.size))
# gfps7 = np.zeros((len(subjlist),t_full.size))

# picks = [4, 25, 30, 31]
# # picks = [31]

# for subj in range(len(subjlist)):
#     sub = subjlist [subj]
#     dat = io.loadmat(save_mat_loc + sub + '_0.5-40Hz_Evoked_AllChan.mat', squeeze_me=True)
#     dat.keys()
    # gfps0[subj,:] = evkd0[0:31].std(axis=0)     #Taking the SD of the first 32 channels, excluding the EXGs
    # gfps1[subj,:] = evkd1[0:31].std(axis=0)
    # gfps2[subj,:] = evkd2[0:31].std(axis=0)
    # gfps3[subj,:] = evkd3[0:31].std(axis=0)
    # gfps4[subj,:] = evkd4[0:31].std(axis=0)
    # gfps5[subj,:] = evkd5[0:31].std(axis=0)
    # gfps6[subj,:] = evkd6[0:31].std(axis=0)
    # gfps7[subj,:] = evkd7[0:31].std(axis=0)


### Sum of squares -- Not doing X-Mean, as 'mean' here is the evoked response here is already referenced to the earlobes
# picks_ss = [3, 30, 26, 4, 25, 7, 31, 22, 8, 21, 11, 12, 18]

# ss_all12 = np.zeros((len(subjlist),t_full.size))
# ss_all20 = np.zeros((len(subjlist),t_full.size))

# for subj in range(len(subjlist)):
#     sub = subjlist [subj]
#     dat = io.loadmat(save_mat_loc + sub + '_1-40Hz_Evoked_AllChan.mat', squeeze_me=True)
#     dat.keys()
#     ss6 = dat['evkd6'][picks_ss]      ###Doing this only for the 5 sec plot
#     ss7 = dat['evkd7'][picks_ss]
#     ss_all12[subj,:] = (ss6**2).sum(axis=0)
#     ss_all20[subj,:] = (ss7**2).sum(axis=0)


#%% GFP | Plotting and saving of 32 channel GFP across all subjects -- Baselined for 1 sec interval for coh and incoherent periods

# fig, ax = plt.subplots(3, 1, sharex = True, constrained_layout=True)
# ax[0].errorbar(t, gfps0.mean(axis=0), yerr=sem(gfps0), label = '12 tone coherence (N='+ str(len(subjlist)) +')', color='green', linewidth=2, ecolor='darkseagreen')
# ax[0].errorbar(t, gfps1.mean(axis=0), yerr=sem(gfps1), label = '20 tone coherence (N='+ str(len(subjlist)) +')', color='purple', linewidth=2, ecolor='thistle')
# ax[1].errorbar(t, gfps2.mean(axis=0), yerr=sem(gfps2), color='green', linewidth=2, ecolor='darkseagreen')
# ax[1].errorbar(t, gfps3.mean(axis=0), yerr=sem(gfps3), color='purple', linewidth=2, ecolor='thistle')
# ax[2].errorbar(t, gfps4.mean(axis=0), yerr=sem(gfps4), color='green', linewidth=2, ecolor='darkseagreen')
# ax[2].errorbar(t, gfps5.mean(axis=0), yerr=sem(gfps5), color='purple', linewidth=2, ecolor='thistle')
# plt.suptitle('Binding - GFP (N='+ str(len(subjlist)) +')')
# fig.subplots_adjust(top=0.88)
# ax[0].set_title('Onset')
# ax[1].title.set_text('Incoherent to Coherent')
# ax[2].title.set_text('Coherent to Incoherent')
# ax[0].legend()
# # fig.tight_layout()
# fig.text(0, 0.55,'Global Field Power(\u03bcV)',fontsize=14, va='center', rotation='vertical')
# plt.xlabel('Time(s)',fontsize=14)
# plt.rcParams["figure.figsize"] = (6.5,5)
# plt.xticks(fontsize=14)
# plt.show()

# ### Calculating DC shift from 300-800 ms (GFP)

# t1 = t>=0.3
# t2 = t<=0.8
# t3 = np.array([t2[i] and t1[i] for i in range(len(t1))])

# gfp12_coh = gfps2[:,t3]
# gfp12_incoh = gfps4[:,t3]
# gfp20_coh = gfps3[:,t3]
# gfp20_incoh = gfps5[:,t3]

# gfp12 = (gfp12_coh - gfp12_incoh).mean(axis=1)         #Calculating coherent - incoherent (kinda baselining)
# gfp20 = (gfp20_coh - gfp20_incoh).mean(axis=1)

# plt.bar(('12','20'), (gfp12.mean(), gfp20.mean()))
# plt.show()

# mat_id = dict(sub=subjlist,gfp12 = gfp12, gfp20=gfp20,gfp12_coh=gfp12_coh,gfp12_incoh=gfp12_incoh,
#               gfp20_coh=gfp20_coh,gfp20_incoh=gfp20_incoh)

# savemat(save_mat_loc + 'AllSubj_GFPDiff_0.4-40Hz_1sec(N=39).mat', mat_id)

#%% GFP | Calculation and saving of 32 channel DC shift in GFP across all subjects -- For entire 5 second duration

# plt.errorbar(t_full, gfps6.mean(axis=0), yerr=sem(gfps6), label = '12 tone coherence (N='+ str(len(subjlist)) +')', color='#1b9e77', linewidth=2, ecolor='#1b9e77',alpha=0.3)
# plt.errorbar(t_full, gfps7.mean(axis=0), yerr=sem(gfps7), label = '20 tone coherence (N='+ str(len(subjlist)) +')', color='#d95f02', linewidth=2, ecolor='#d95f02',alpha=0.3)
# plt.suptitle('Binding (N='+ str(len(subjlist)) +')')
# plt.subplots_adjust(top=0.88)
# plt.legend()
# # fig.tight_layout()
# plt.ylabel('Global Field Power(\u03bcV)')
# plt.xlabel('Time(s)',fontsize=14)
# plt.rcParams["figure.figsize"] = (6.5,5)
# plt.xticks(fontsize=14)
# plt.show()

# ### Calculating DC shift from 300-800 ms (GFP)
# t1 = t_full>=0.3
# t2 = t_full<=0.8
# t3 = np.array([t2[i] and t1[i] for i in range(len(t1))])
# t4 = t_full>=1.3
# t5 = t_full<=1.8
# t6 = (np.array([t4[i] and t5[i] for i in range(len(t4))]))
# t7 = t_full>=2.3
# t8 = t_full<=2.8
# t9 = np.array([t7[i] and t8[i] for i in range(len(t7))])
# t10 = t_full>=3.3
# t11 = t_full<=3.8
# t12 = np.array([t10[i] and t11[i] for i in range(len(t10))])
# t13 = t_full>=4.3
# t14 = t_full<=4.8
# t15 = np.array([t13[i] and t14[i] for i in range(len(t13))])

# gfp12_1=gfps6[:,t3]
# gfp12_2=gfps6[:,t6]
# gfp12_3=gfps6[:,t9]
# gfp12_4=gfps6[:,t12]
# gfp12_5=gfps6[:,t15]

# gfp20_1=gfps7[:,t3]
# gfp20_2=gfps7[:,t6]
# gfp20_3=gfps7[:,t9]
# gfp20_4=gfps7[:,t12]
# gfp20_5=gfps7[:,t15]

# gfp12_coh = gfp12_2 + gfp12_4
# gfp12_incoh =  gfp12_3 + gfp12_5
# gfps12_5sec=(gfp12_coh-gfp12_incoh).mean(axis=1)

# gfp20_coh = gfp20_2 + gfp20_4
# gfp20_incoh =  gfp20_3 + gfp20_5
# gfps20_5sec=(gfp20_coh-gfp20_incoh).mean(axis=1)

# plt.bar(('12','20'), (gfps12_5sec.mean(), gfps20_5sec.mean()))
# plt.show()

# mat_ids1=dict(subj=subjlist,gfps12_5sec=gfps12_5sec, gfps20_5sec=gfps20_5sec)

# savemat(save_mat_loc + 'AllSubj_GFPDiff_0.4-40Hz_5sec(N=39)_New.mat', mat_ids1)

#%% SS | Plotting sum of squares across all subjects -- Baselined for 1 sec interval for coh and incoherent periods

# plt.errorbar(t_full, ss_all12.mean(axis=0), yerr=sem(ss_all12), label = '12 tone coherence (N='+ str(len(subjlist)) +')', color='green', linewidth=2, ecolor='darkseagreen')
# plt.errorbar(t_full, ss_all20.mean(axis=0), yerr=sem(ss_all20), label = '20 tone coherence (N='+ str(len(subjlist)) +')', color='purple', linewidth=2, ecolor='thistle')
# plt.suptitle('Binding - Sum of Squares (N='+ str(len(subjlist)) +')')
# plt.subplots_adjust(top=0.88)
# plt.legend()
# # fig.tight_layout()
# plt.ylabel('Sum of Squares (\u03bcV)')
# plt.xlabel('Time(s)',fontsize=14)
# plt.rcParams["figure.figsize"] = (6.5,5)
# plt.xticks(fontsize=14)
# plt.show()