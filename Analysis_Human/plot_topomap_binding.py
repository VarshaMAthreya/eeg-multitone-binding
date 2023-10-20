# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:48:08 2023

@author: vmysorea
"""
import numpy as np
import sys
sys.path.append('C:/Users/vmysorea/Documents/mne-python/')
import mne
from mne.channels.layout import _pol_to_cart, _cart_to_sph
from matplotlib import pyplot as plt
from scipy import io

montage = mne.channels.make_standard_montage('biosemi32')

chs = montage._get_ch_pos()
ch_names, xyz = zip(*[(ich, ixyz) for ich, ixyz in chs.items()])
xyz = np.vstack(xyz)

mne.viz.plot_montage(montage)
sph = _cart_to_sph(xyz)
xy = _pol_to_cart(sph[:, 1:][:, ::-1]) * 0.05

#Calculating my evoked array
save_loc='C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/Binding/Human_Analysis/Figures/'
save_epochs_loc = 'D:/PhD/Data/Epochs-fif/'
save_mat_loc = 'D:/PhD/Data/Binding_matfiles/'

subjlist = ['S273','S268','S269','S274','S282','S285','S259','S277','S279','S280','S270','S271','S281','S290','S284',
              'S303','S288','S260','S341','S312','S347','S340','S078','S069', 'S088','S072','S308','S344','S105','S291','S310','S339']
evokeds = []

evokeds_y = []
evokeds_o =[]
# evoked=a, evoked1 = b, evoked2 = c, evoked3 = c,fs=4096, t=epochs.times
gfps20_y = np.zeros((len(subjlist),22529))
evk_y = np.zeros((len(subjlist),22529))

for subj in range(len(subjlist)):
    sub = subjlist[subj]
    dat = io.loadmat(save_mat_loc + sub + '_allevoked0.4.mat', squeeze_me=True)
    dat.keys()
    evoked_y = dat['evoked']
    # evoked1_y = dat['evoked1']
    # evoked2_y = dat['evoked2']
    # evoked3_y = dat['evoked3']
    fs = dat['fs']
    t = dat['t']  
    gfp20_y=evoked_y.std(axis=0)   
    gfps20_y[subj,:]=gfp20_y
    evokeds_y += [evoked_y,]
    y = evoked_y.mean(axis=0)
    # # y1 = evoked1_y.mean(axis=0)
    # # y2 = evoked2_y.mean(axis=0)
    # # y3 = evoked3_y.mean(axis=0)
    # evk_y[subj,:] = y
    # evk_y1[subj,:] = y1
    # evk_y2[subj,:] = y2
    # evk_y3[subj,:] = y3

t=t
# t=np.array(t*4096)

t1 = t>=0.5
t2 = t<=0.8
t3 = np.array([t2[i] and t1[i] for i in range(len(t1))])
t4 = t>=1.
t5 = t<=1.3
t6 = (np.array([t4[i] and t5[i] for i in range(len(t4))]))
t7 = t>=1.5
t8 = t<=1.8
t9 = np.array([t7[i] and t8[i] for i in range(len(t7))])
# t10 = t>=3.3
# t11 = t<=3.8
# t12 = np.array([t10[i] and t11[i] for i in range(len(t10))])
# t13 = t>=4.3
# t14 = t<=4.8
# t15 = np.array([t13[i] and t14[i] for i in range(len(t13))])

    
evoked_1=np.zeros((32,len(t3)))
evoked_2=np.zeros((32,len(t6)))
evoked_3=np.zeros((32,len(t9)))

for c in range(len(subjlist)):
    evoked_1=evokeds_y[c][:,t3]
    evoked_2=evokeds_y[c][:,t6]
    evoked_3=evokeds_y[c][:,t9]

evoked = evoked_1 + evoked_2+evoked_3
# evoked_incoh =  evoked_3 + evoked_5
# evoked_topo=evoked_coh-evoked_incoh


# Replace this with difference calculated from 5-second evoked response
array_to_plot_as_topomap = evoked.mean(axis=1)

fig=mne.viz.plot_topomap(array_to_plot_as_topomap, xy, contours=15, res=128, size=4.5)
# plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.88)
plt.tight_layout()
plt.rcParams['axes.titlepad'] = 0
plt.title('Across Age', fontsize=14,loc='center')

plt.savefig(save_loc + 'GDT_All_Topomap', dpi=300)

############Binding 
for subj in subjlist:
    evoked= mne.read_evokeds(save_epochs_loc + subj +'_Binding_evoked20_0.4-ave.fif',baseline=(-0.3,0))    
    evokeds+=[evoked,]

for c in range(len(subjlist)):
    evokeds_all = mne.combine_evoked(evokeds[c], weights='equal')
    
t=evokeds_all.times
# t=np.array(t*4096)

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


evoked_1=np.zeros((32,len(t3)))
evoked_2=np.zeros((32,len(t6)))
evoked_3=np.zeros((32,len(t9)))
evoked_4=np.zeros((32,len(t12)))
evoked_5=np.zeros((32,len(t15)))

evoked_1=evokeds_all.data[:,t3]
evoked_2=evokeds_all.data[:,t6]
evoked_3=evokeds_all.data[:,t9]
evoked_4=evokeds_all.data[:,t12]
evoked_5=evokeds_all.data[:,t15]

evoked_coh = evoked_2 + evoked_4
evoked_incoh =  evoked_3 + evoked_5
evoked_topo=evoked_coh-evoked_incoh


# Replace this with difference calculated from 5-second evoked response
array_to_plot_as_topomap = evoked_topo.mean(axis=1)

fig=mne.viz.plot_topomap(array_to_plot_as_topomap, xy, contours=15, res=128, size=4.5)
# plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.88)
plt.tight_layout()
plt.rcParams['axes.titlepad'] = 0
plt.title('All Subjects', fontsize=14,loc='center')

plt.savefig(save_loc + 'AllBinding_Topomap', dpi=300)
