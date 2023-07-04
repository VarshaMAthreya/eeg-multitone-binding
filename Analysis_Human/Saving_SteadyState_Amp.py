# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:23:11 2023

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
save_mat_loc = 'D:/PhD/Data/Binding_matfiles/'

### Haven't considered S273 and S345
subjlist = ['S268', 'S269', 'S274', 'S282', 'S285',
            'S277', 'S279', 'S280','S259','S270', 'S271', 
            'S281','S290', 'S284', 'S305','S303','S288','S260',
            'S352', 'S341', 'S312', 'S347', 'S340','S078','S069',
            'S088','S342','S072','S308','S344','S291','S310','S339']
# subjlist = ['S273', 'S268', 'S269', 'S274', 'S282', 'S285',
           # 'S277', 'S279', 'S280','S259','S270', 'S271', 
           # 'S281','S290', 'S284', 'S305','S303','S288','S260',
           # 'S352', 'S341', 'S312', 'S347', 'S340','S078','S069',
           # 'S088','S342','S072','S308','S344','S345','S291','S310','S339']

subjlist_y = ['S273','S268', 'S269', 'S274', 'S282', 'S285',
            'S277', 'S279', 'S280','S259','S270', 'S271', 
            'S281','S290', 'S284', 'S305','S303','S288','S260',
            'S352', 'S341']

subjlist_o = ['S312', 'S347', 'S340','S078','S069', 
              'S088','S342','S072','S308','S344','S105',
              'S291','S310','S339']
