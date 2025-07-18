#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:38:21 2024

@author: brandon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial


def bin_aps(df, bins, ap_col='y'):
    _counts, bins = np.histogram(df.get(ap_col), bins)
    bins = bins[1:]
    partial_func = partial(get_ap_bin, bins=bins)
    binned_aps = df.get(ap_col).apply(partial_func)
    df[f'binned_{ap_col}'] = binned_aps.values
    
    return df
    

def get_ap_bin(this_ap, bins):
    this_bin = np.where(np.abs(this_ap - bins) == np.nanmin(np.abs(this_ap - bins)))[0][0]

    return this_bin
    


df = pd.read_pickle(r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_03_20_dpt-gfp_r4-gal4_ecoli-hs-dtom_input-output_pilot_4-6hrs/larva_5/bacteria_quant.pkl')

"""bin ap"""
n_ap_bins = 22
ap_bins = np.linspace(0, 1, n_ap_bins + 1)

df['ap'] = df.y / df.y.max()
df = bin_aps(df, ap_bins, ap_col='ap')
ap_col = 'ap'

method = 'bkg_sub_sum_data'

"""plot histogram of intensities"""
intens = df.get(method)
n_bins = 50
bins = np.logspace(np.log10(np.min(intens)), np.log10(np.max(intens)), n_bins)
counts, _ = np.histogram(intens, bins)

plt.figure()
plt.plot(bins[:-1], counts, 'ko', markersize=16, markerfacecolor='none', markeredgewidth=3)
plt.xscale("log")
plt.yscale("log")

single_cell_inten = np.median(intens)
scaled_intens = intens / single_cell_inten
scaled_intens[scaled_intens < 1] = 1
scaled_intens = scaled_intens.astype('uint32')

bins = np.logspace(np.log10(np.min(scaled_intens)), np.log10(np.max(scaled_intens)), n_bins)
counts, _ = np.histogram(scaled_intens, bins)

plt.figure()
plt.plot(bins[:-1], counts, 'ko', markersize=16, markerfacecolor='none', markeredgewidth=3)
plt.xscale("log")
plt.yscale("log")

"""plot number of bacteria vs ap axis"""
binned_sum = df.get([method, f'binned_{ap_col}']).groupby(by=f'binned_{ap_col}').sum().values.flatten()

ap = ap_bins[:-1] / np.max(ap_bins[:-1])
plt.figure()
plt.plot(ap, binned_sum, 'm-', linewidth=4)
#plt.yscale('log')