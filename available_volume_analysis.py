#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 13:05:18 2024

@author: brandon
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from functools import partial

    

def get_ap_bin(this_ap, bins):
    this_bin = np.where(np.abs(this_ap - bins) == np.nanmin(np.abs(this_ap - bins)))[0][0]

    return this_bin


experiment_paths = [r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_06_26-NP1029-Gal4_UAS-Mhc-RNAi_ecoli-hs-dtom_4hrs/no_heartbeat']
#experiment_paths = [r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_07_02-NP1029-Gal4_ecoli-hs-dtom_2hr']

n_ap_bins = 22
ap_bins = np.linspace(0, 1, n_ap_bins)

avs = []
for path in experiment_paths:
    larvae_dirs = glob(path + '/larva*')
    for larvae_dir in larvae_dirs:
        df = pd.read_pickle(larvae_dir + '/available_pixels.pkl')
        avs.append(df.available_pixels.values)
        

av_arr = np.zeros((len(avs), n_ap_bins))
for j, av in enumerate(avs):
    x = np.linspace(0, 1, len(av))
    
    for i in range(len(av)):
        this_bin = get_ap_bin(x[i], ap_bins)
        av_arr[j, this_bin] += av[i]

plt.figure()
# for av in avs:
#     plt.plot(np.flip(av))
x = np.linspace(0, 1, n_ap_bins)
for av in av_arr:
    plt.plot(x, av)
    
plt.figure()
m = np.mean(av_arr, axis=0)
s = np.std(av_arr, axis=0)
l = m - s
u = m + s
color = [0.5, 0.5, 0.5]
plt.fill_between(x, l, u, color=color, alpha=0.5)
plt.plot(x, m, color=color, linewidth=4)