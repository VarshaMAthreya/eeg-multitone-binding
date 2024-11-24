# -*- coding: utf-8 -*-
"""

Created on Fri Aug 20 15:49:37 2021

@author: ravinderjit

"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from anlffr.preproc import find_blinks
from EEGpp import EEGconcatenateFolder
from mne.preprocessing.ssp import compute_proj_epochs
import os
import pickle
#from sklearn.decomposition import PCA

nchans = 34;
refchans = ['EXG1','EXG2']


Subjects = [ 'S069', 'S072','S078','S088', 'S104', 'S105', 'S207','S211',
            'S259', 'S260', 'S268', 'S269', 'S270','S271', 'S272', 'S273',
            'S274', 'S277','S279', 'S280', 'S282', 'S284', 'S285', 'S288',
            'S290' ,'S281','S291', 'S303', 'S305', 'S308', 'S309', 'S310',
            'S312', 'S339', 'S340', 'S341', 'S344', 'S345', 'S337', 'S352',
            'S355', 'S358']


fig_loc =  '/media/ravinderjit/Data_Drive/Data/Figures/MTBproj/Binding/'
data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/MTB/Binding'
pickle_loc = data_loc + '/Pickles/'
exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved


for subject in Subjects:

    datapath = os.path.join(data_loc,subject + '_Binding')

    if subject == 'S337':
        refchans = ['A7', 'A24']
        data_eeg,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans)

    data_eeg,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
    data_eeg.filter(l_freq=0.5,h_freq=40)

    if subject == 'S273':
        data_eeg.info['bads'].append('A1')
        data_eeg.info['bads'].append('A30')
        data_eeg.info['bads'].append('A24')

    if subject == 'S271':
        data_eeg.info['bads'].append('A3')
        data_eeg.info['bads'].append('A6')
        data_eeg.info['bads'].append('A16')
        data_eeg.info['bads'].append('A1')
        data_eeg.info['bads'].append('A30')

    if subject == 'S284':
        data_eeg.info['bads'].append('A6')
        data_eeg.info['bads'].append('A24')
        data_eeg.info['bads'].append('A25')
        data_eeg.info['bads'].append('A28')
        data_eeg.info['bads'].append('A3')

    if subject == 'S072':
        data_eeg.info['bads'].append('A1')
        data_eeg.info['bads'].append('A30')

    if subject == 'S088':
        data_eeg.info['bads'].append('A6') #not that bad

    if subject == 'S303':
        data_eeg.info['bads'].append('A1') #not that bad
        data_eeg.info['bads'].append('A30') #not that bad

    if subject == 'S069':
        data_eeg.info['bads'].append('A24')

    if subject == 'S104':
        data_eeg.info['bads'].append('A6')
        data_eeg.info['bads'].append('A8')
        data_eeg.info['bads'].append('A18')
        data_eeg.info['bads'].append('A19')
        data_eeg.info['bads'].append('A24')
        data_eeg.info['bads'].append('A25')

    if subject == 'S309':
        data_eeg.info['bads'].append('A6')

    if subject == 'S312':
        data_eeg.info['bads'].append('A3')
        data_eeg.info['bads'].append('A28')
        data_eeg.info['bads'].append('A6')
        data_eeg.info['bads'].append('A8')
        data_eeg.info['bads'].append('A23')

    if subject == 'S340':
        data_eeg.info['bads'].append('A20')
        data_eeg.info['bads'].append('A30')
        data_eeg.info['bads'].append('A24')
        data_eeg.info['bads'].append('A28')


    if subject == 'S341':
        data_eeg.info['bads'].append('A7')
        data_eeg.info['bads'].append('A20')


    if subject == 'S345':
        data_eeg.info['bads'].append('A6')
        data_eeg.info['bads'].append('A3')
        data_eeg.info['bads'].append('A18')

    if subject == 'S337':
        data_eeg.info['bads'].append('A21')
        data_eeg.info['bads'].append('A1')
        data_eeg.info['bads'].append('A30')
        data_eeg.info['bads'].append('A2')
        data_eeg.info['bads'].append('A29')

    if subject == 'S352':
        data_eeg.info['bads'].append('A17')
        data_eeg.info['bads'].append('A9')





    #%% Remove Blinks

    blinks = find_blinks(data_eeg,ch_name = ['A1'],thresh = 100e-6, l_trans_bandwidth = 0.5, l_freq =1.0)
    blink_epochs = mne.Epochs(data_eeg,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                                  baseline=(-0.25,0),reject=dict(eeg=500e-6))
    Projs = compute_proj_epochs(blink_epochs,n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')

    ocular_projs = [Projs[0]]

    data_eeg.add_proj(ocular_projs)
    data_eeg.plot_projs_topomap()
    plt.savefig(os.path.join(fig_loc,'OcularProjs',subject + '_OcularProjs.png'),format='png')
    #plt.close()
    #data_eeg.plot(events=blinks,show_options=True)

    #%% Add events for AB transitions at t = 1,2,3,4

    data_evnt_AB = data_evnt.copy()
    fs = data_eeg.info['sfreq']

    for cnd in range(2):
        for e in range(4):
            evnt_num = 3 + e + cnd*4
            events_add = data_evnt[data_evnt[:,2] == int(cnd+1),:] + [int(fs*(e+1)),int(0),evnt_num - (cnd+1)]
            data_evnt_AB = np.concatenate((data_evnt_AB,events_add),axis=0)



    #%% Plot Data

    conds = ['12','20'] #14,18 for S211 from earlier date
    reject = dict(eeg=150e-6)
    epochs_whole = []
    evkd_whole = []

    for cnd in range(len(conds)):
        ep_cnd = mne.Epochs(data_eeg,data_evnt,cnd+1,tmin=-0.3,tmax=5.3,reject = reject, baseline = (-0.1,0.))
        epochs_whole.append(ep_cnd)
        evkd_whole.append(ep_cnd.average())
        evkd_whole[cnd].plot(titles=conds[cnd])

    #%% Extract Different Conditions

    conds = ['12_0', '20_0', '12_AB1', '12_BA1', '12_AB2', '12_BA2', '20_AB1','20_BA1','20_AB2','20_BA2']

    epochs = []
    evkd = []
    for cnd in range(10):
        ep_cnd = mne.Epochs(data_eeg,data_evnt_AB,cnd+1,tmin=-0.2,tmax=1.1, reject = reject, baseline = (-0.1,0.))
        epochs.append(ep_cnd)
        evkd.append(ep_cnd.average())
        #evkd[cnd].plot(picks=31,titles=conds[cnd])

    conds.extend(['12AB', '12BA','20AB','20BA'])
    ev_combos = [[2,4],[3,5],[6,8],[7,9]]

    for it, cnd in enumerate(range(10,14)):
        ep_cnd = mne.Epochs(data_eeg,data_evnt_AB,list(np.array(ev_combos[it])+1),tmin=-0.2,tmax=1.1, reject = reject, baseline = (-0.1,0.))
        epochs.append(ep_cnd)
        evkd.append(ep_cnd.average())
        #evkd[cnd].plot(picks=31,titles=conds[cnd])

    # Also get whole interval without baselining each interval
    conds.extend(['12', '20' ])
    ep_cnd = mne.Epochs(data_eeg,data_evnt,1,tmin=-0.3,tmax=5.5, reject = reject, baseline = (-0.1,0.))
    epochs.append(ep_cnd)
    evkd.append(ep_cnd.average())

    ep_cnd = mne.Epochs(data_eeg,data_evnt,2,tmin=-0.3,tmax=5.5, reject = reject, baseline = (-0.1,0.))
    epochs.append(ep_cnd)
    evkd.append(ep_cnd.average())



    #%% Plot 1st and second interval

    for it, c in enumerate(ev_combos):
        evkds = [evkd[c[0]], evkd[c[1]]]
        #mne.viz.plot_compare_evokeds(evkds,picks=31,title = conds[it + 10])


    #%% Plot Comparisons

    combos_comp = [[0,1], [10,12], [11,13]]
    comp_labels = ['Onset', 'Incoherent to Coherent', 'Coherent to Incoherent']

    for it,c in enumerate(combos_comp):
        evkds = [evkd[c[0]], evkd[c[1]]]
        #mne.viz.plot_compare_evokeds(evkds,title=comp_labels[it])


    #%% Make Plots outside of MNE

    fig, ax = plt.subplots(3,1,sharex=True)

    t = epochs[0].times
    for cnd in range(len(combos_comp)):
        cz_ep_12 = epochs[combos_comp[cnd][0]].get_data()[:,31,:]
        cz_mean_12 = cz_ep_12.mean(axis=0)
        cz_sem_12 = cz_ep_12.std(axis=0) / np.sqrt(cz_ep_12.shape[0])

        cz_ep_20 = epochs[combos_comp[cnd][1]].get_data()[:,31,:]
        cz_mean_20 = cz_ep_20.mean(axis=0)
        cz_sem_20 = cz_ep_20.std(axis=0) / np.sqrt(cz_ep_20.shape[0])

        ax[cnd].plot(t,cz_mean_12,label='12')
        ax[cnd].fill_between(t,cz_mean_12 - cz_sem_12, cz_mean_12 + cz_sem_12,alpha=0.5)

        ax[cnd].plot(t,cz_mean_20,label='20')
        ax[cnd].fill_between(t,cz_mean_20 - cz_sem_20, cz_mean_20 + cz_sem_20,alpha=0.5)

        ax[cnd].set_title(comp_labels[cnd])
        ax[cnd].ticklabel_format(axis='y',style='sci',scilimits=(0,0))

    ax[0].legend()
    ax[2].set_xlabel('Time(sec)')
    ax[2].set_ylabel('Amplitude \$uV')

    plt.savefig(os.path.join(fig_loc,subject + '_12vs20.png'),format='png')
    #plt.close()



    #%% Compute induced activity

    # freqs = np.arange(1.,90.,1.)
    # T = 1./5
    # n_cycles = freqs*T
    # time_bandwidth = 2
    # vmin = -.15
    # vmax = abs(vmin)
    # bline = (-0.1,0)

    # channels = np.arange(32)

    # tfr_12 = mne.time_frequency.tfr_multitaper(epochs_whole[0].subtract_evoked(),
    #                                            freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth,
    #                                            return_itc = False,picks = channels,decim=4)#,average=False)

    # tfr_20 = mne.time_frequency.tfr_multitaper(epochs_whole[1].subtract_evoked(),
    #                                            freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth,
    #                                            return_itc = False,picks = channels,decim=4)#,average=False)

    # tfr_12.plot_topo(baseline =bline,mode= 'logratio', title = '12', vmin=vmin,vmax=vmax)

    # tfr_20.plot_topo(baseline =bline,mode= 'logratio', title = '20', vmin=vmin,vmax=vmax)


    #%% Save Epochs

    save_indexes = [0,1,10,11,12,13,14,15]
    conds_save = []
    epochs_save = []
    evkd_save = [] #save 32 channel evkd response

    t_full = epochs[-1].times

    for si in save_indexes:
        conds_save.append(conds[si])
        evkd_save.append(evkd[si])
        epochs_save.append(epochs[si].get_data()[:,31,:]) # Only save epochs for channel 31

    with open(os.path.join(pickle_loc,subject+'_Binding.pickle'),'wb') as file:
        pickle.dump([t, t_full, conds_save, epochs_save,evkd_save],file)

    del epochs, evkd, evkd_save,epochs_save