#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:49:39 2024

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


#experiment_paths = [r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_06_26-NP1029-Gal4_UAS-Mhc-RNAi_ecoli-hs-dtom_4hrs/no_heartbeat']
experiment_paths = [r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_07_02-NP1029-Gal4_ecoli-hs-dtom_2hr']

n_ap_bins = 22
ap_bins = np.linspace(0, 1, n_ap_bins)

widths = []
for path in experiment_paths:
    larvae_dirs = sorted(glob(path + '/larva*'))
    for larvae_dir in larvae_dirs:
        if len(glob(larvae_dir + '/width.pkl')) > 0:
            print(larvae_dir)
            df = pd.read_pickle(larvae_dir + '/width.pkl')
            these_widths = df.width.values
            first_id = np.where(these_widths > 5)[0][0]
            last_id = np.where(these_widths > 5)[0][-1]
            these_widths = these_widths[first_id:last_id]
            widths.append(these_widths)
        

widths_arr = np.zeros((len(widths), n_ap_bins))
for j, width in enumerate(widths):
    x = np.linspace(0, 1, len(width))
    
    these_bins = np.zeros_like(width)
    for i in range(len(width)):
        this_bin = get_ap_bin(x[i], ap_bins)
        these_bins[i] = this_bin
    
    for i in range(len(ap_bins)):
        widths_arr[j, i] = np.mean(width[these_bins == i])

plt.figure()
# for av in avs:
#     plt.plot(np.flip(av))
x = np.linspace(0, 1, n_ap_bins)
for i, width in enumerate(widths_arr):
    plt.plot(x, width, label=str(i))
plt.legend()
    
plt.figure()
m = np.mean(widths_arr, axis=0)
s = np.std(widths_arr, axis=0)
l = m - s
u = m + s
color = [0.5, 0.5, 0.5]
plt.fill_between(x, l, u, color=color, alpha=0.5)
plt.plot(x, m, color=color, linewidth=4)

mean_width_um = m * 0.91 * 20
inner_radius = np.clip(mean_width_um / 2 - 65, a_min=0, a_max=np.inf)
cross_sectional_area = 4 * np.pi * ((mean_width_um / 2) **2 - inner_radius ** 2)
volume = cross_sectional_area * 3000 / n_ap_bins
volume