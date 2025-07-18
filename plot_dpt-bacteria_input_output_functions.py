#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:04:49 2024

@author: brandon
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from glob import glob
from scipy.ndimage import gaussian_filter


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
    


def compute_larva_width_from_mip(mip, sigma_blur, thresh, dxy=1.0):
    #mip_max = np.max(mip)
    mip = gaussian_filter(mip, sigma=sigma_blur)
    mip = mip > thresh

    larva_width = np.zeros(len(mip))
    for i in range(len(mip)):
        ids = np.where(mip[i])[0]
        if len(ids) > 1:
            start_id = ids[0]
            end_id = ids[-1]
    
            larva_width[i] = (end_id - start_id) * dxy
        else:
            larva_width[i] = 0

    return larva_width



def compute_dpt_line_dist_from_mip(mip, thresh=240):
    mip[mip < thresh] = 0
    line_dist = np.sum(mip.astype('float'), axis=1)
    
    return line_dist


experiment_paths = [r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_03_20_dpt-gfp_r4-gal4_ecoli-hs-dtom_input-output_pilot_4-6hrs']
df_name = 'bacteria.pkl'
method = 'bkg_sub_sum_data'

sigma_blur = 3
thresh = 360
dxy = 0.325

dpt_thresh = 240
n_ap_bins = 22
ap_bins = np.linspace(0, 1, n_ap_bins + 1)
#plt.figure()
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

for path in experiment_paths:
    larvae_dirs = glob(path + '/larva*')
    for larvae_dir in larvae_dirs:
        if larvae_dir == r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_03_20_dpt-gfp_r4-gal4_ecoli-hs-dtom_input-output_pilot_4-6hrs/larva_1':
            continue
        df = pd.read_pickle(larvae_dir + '/' + df_name)

        df['ap'] = df.y / df.y.max()
        df = bin_aps(df, ap_bins, ap_col='ap')
        ap_col = 'ap'
        ap_bin_width_microns = df.y.max() * dxy / n_ap_bins

        intens = df.get(method)
        single_cell_inten = np.median(intens)
        df['n_bacteria'] = np.clip(intens / single_cell_inten, a_min=1, a_max=np.inf).astype('int')
        
        df = df[df.n_bacteria < 3]
        """plot number of bacteria vs ap axis"""
        binned_sum = df.get(['n_bacteria', f'binned_{ap_col}']).groupby(by=f'binned_{ap_col}').sum().values.flatten()
        ap_bin_numbers = sorted(df.get(f'binned_{ap_col}').unique())

        # normalize by larva cross sectional area
        mip = plt.imread(larvae_dir + '/mips/mip_red_0.tif')
        larva_width = compute_larva_width_from_mip(mip, sigma_blur, thresh, dxy=dxy)
        
        binned_larva_width = np.zeros(len(ap_bin_numbers))
        y_pixels = np.arange(mip.shape[0])
        for i in range(len(ap_bin_numbers)):
            sel = np.abs(y_pixels / np.max(y_pixels) - ap_bins[ap_bin_numbers[i]]) == np.min(np.abs(y_pixels / np.max(y_pixels) - ap_bins[ap_bin_numbers[i]]))
            if np.sum(sel) > 1:
                sel[np.where(sel)[0][-1]] = False
            binned_larva_width[i] = larva_width[sel]
        
        ap_bin_volume = (4 * np.pi / 3) * (binned_larva_width / 2) ** 2 * ap_bin_width_microns
        binned_concentration = binned_sum / ap_bin_volume * 10 ** 9
        
        ap = ap_bins[:-1] / np.max(ap_bins[:-1])
        
        """compute dpt_line dist and bin"""
        dpt_mip = plt.imread(larvae_dir + '/mips_crop_gut/mip_crop_gut_channel0_t0.tif')
        dpt_line_dist = compute_dpt_line_dist_from_mip(dpt_mip, thresh=dpt_thresh)
        binned_dpt_line_dist = np.zeros(len(ap_bin_numbers))
        y_pixels = np.arange(mip.shape[0])
        for i in range(len(ap_bin_numbers)):
            sel = np.abs(y_pixels / np.max(y_pixels) - ap_bins[ap_bin_numbers[i]]) == np.min(np.abs(y_pixels / np.max(y_pixels) - ap_bins[ap_bin_numbers[i]]))
            if np.sum(sel) > 1:
                sel[np.where(sel)[0][-1]] = False
            binned_dpt_line_dist[i] = dpt_line_dist[sel]
        
        ax1.plot(ap, binned_concentration, 'r-', linewidth=4)
        
        
        ax2.plot(ap, binned_dpt_line_dist, 'b-', linewidth=4)
       
        
        #plt.plot(binned_concentration[np.array(ap_bin_numbers) < n_ap_bins / 2], binned_dpt_line_dist[np.array(ap_bin_numbers) < n_ap_bins / 2], 'o', linewidth=4)
        plt.xlabel('fraction of ap axis', fontsize=24)
        #plt.ylabel('dpt-gfp (a.u.)', fontsize=24)
        # plt.plot(ap, binned_dpt_line_dist)
        # #plt.plot(ap, binned_concentration, '-', linewidth=4)
        # #plt.plot(ap, binned_sum, '-', linewidth=4)

        # plt.xlabel('fraction of anterior-posterior axis', fontsize=24)
        # plt.ylabel('dpt-gfp', fontsize=24)
ax1.tick_params(axis='y', labelcolor='r')
ax1.set_label('bacterial concentration (cells/ml^3)')
ax1.set_ylim([0, 75000])
ax2.tick_params(axis='y', labelcolor='b')
ax2.set_label('dpt-gfp (a.u.)')
plt.xlim([0.15, 0.6])