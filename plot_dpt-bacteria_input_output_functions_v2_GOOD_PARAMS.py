#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:41:30 2024

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
    mip = mip.astype("float") - thresh
    mip[mip < 0] = 0
    line_dist = np.sum(mip.astype('float'), axis=1) / np.sum(mip > 0, axis=1)
    
    return line_dist


experiment_paths = [r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_03_20_dpt-gfp_r4-gal4_ecoli-hs-dtom_input-output_pilot_4-6hrs']
df_name = 'bacteria.pkl'
method = 'bkg_sub_sum_data'

sigma_blur = 3
thresh = 360
dxy = 0.325

dpt_thresh = 240
n_ap_bins = 18
ap_bins = np.linspace(0, 1, n_ap_bins + 1)
plt.figure()

all_bacteria = []
all_dpt = []
all_ap = []
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
        
      
        sel = np.array(np.array(ap_bin_numbers) < (0.6 * n_ap_bins)) * np.array(np.array(ap_bin_numbers) > (0.2 * n_ap_bins))
        these_concentrations = binned_concentration[sel]
        these_dpt = binned_dpt_line_dist[sel]
        plt.plot(these_concentrations, these_dpt, '-', linewidth=4)
        #plt.xlabel('fraction of ap axis', fontsize=24)
        #plt.ylabel('dpt-gfp (a.u.)', fontsize=24)
        # plt.plot(ap, binned_dpt_line_dist)
        # #plt.plot(ap, binned_concentration, '-', linewidth=4)
        # #plt.plot(ap, binned_sum, '-', linewidth=4)

        # plt.xlabel('fraction of anterior-posterior axis', fontsize=24)
        # plt.ylabel('dpt-gfp', fontsize=24)
        all_bacteria.append(these_concentrations)
        all_dpt.append(these_dpt)
        all_ap.append(ap[np.array(ap_bin_numbers)[sel]])

all_bacteria_arr = np.zeros((len(all_bacteria), len(all_bacteria[0])))
all_dpt_arr = np.zeros((len(all_dpt), len(all_dpt[0])))
all_ap_arr = np.zeros((len(all_ap), len(all_dpt[0])))

for i in range(len(all_bacteria_arr)):
    all_bacteria_arr[i] = all_bacteria[i]
    all_dpt_arr[i] = all_dpt[i]
    all_ap_arr[i] = all_ap[i]

mean_bacteria = np.mean(all_bacteria_arr, axis=0)
mean_dpt = np.mean(all_dpt_arr, axis=0)
mean_ap = np.mean(all_ap_arr, axis=0)
std_bacteria = np.std(all_bacteria, axis=0)
std_dpt = np.std(all_dpt, axis=0)
std_ap = np.std(all_ap, axis=0)

# plot double y of bacteria concentration and dpt-gfp
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.fill_between(mean_ap, mean_bacteria - std_bacteria, mean_bacteria + std_bacteria, color='m', linewidth=4, alpha=0.3)
ax1.plot(mean_ap, mean_bacteria, 'r-', linewidth=4)
ax2.fill_between(mean_ap, mean_dpt - std_dpt, mean_dpt + std_dpt, color='g', linewidth=4, alpha=0.3)
ax2.plot(mean_ap, mean_dpt, 'g-', linewidth=4)
ax1.tick_params(axis='y', labelcolor='m')
ax1.set_label('bacterial concentration (cells/ml)')
ax1.set_ylim([0, 75000])
ax2.tick_params(axis='y', labelcolor='g')
ax2.set_label('dpt-gfp (a.u.)')
plt.xlim([np.min(mean_ap), np.max(mean_ap)])

# plot bacteria concentration and dpt-gfp
plt.figure()
plt.subplot(211)
plt.fill_between(mean_ap, mean_bacteria - std_bacteria, mean_bacteria + std_bacteria, color='m', linewidth=4, alpha=0.3)
plt.plot(mean_ap, mean_bacteria, 'm-', linewidth=4)
plt.xlim([np.min(mean_ap), np.max(mean_ap)])
plt.ylabel('bacterial concentration (cells/ml)')

plt.subplot(212)
plt.fill_between(mean_ap, mean_dpt - std_dpt, mean_dpt + std_dpt, color='g', linewidth=4, alpha=0.3)
plt.plot(mean_ap, mean_dpt, 'g-', linewidth=4)
plt.ylabel('dpt-gfp (a.u.)')
plt.xlim([np.min(mean_ap), np.max(mean_ap)])



sorted_ids = np.argsort(mean_bacteria)
mean_bacteria = mean_bacteria[sorted_ids]
mean_dpt = mean_dpt[sorted_ids]
std_bacteria = std_bacteria[sorted_ids]
std_dpt = std_dpt[sorted_ids]

plt.figure()
#plt.plot(mean_bacteria, mean_dpt, 'ko', linewidth=4)
plt.errorbar(mean_bacteria, mean_dpt, yerr=std_dpt, xerr=std_bacteria, 
             ecolor='k', elinewidth=3, capsize=4, marker='o', 
             markeredgewidth=2, markeredgecolor='k', markersize=18,
             markerfacecolor='c', alpha=0.7, linewidth=4, color='c')




