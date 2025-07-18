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
from matplotlib import rc
from scipy.optimize import curve_fit
import pickle



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


def style_axes(ax, fontsize=24):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    
    return ax


def hill(x, A, n, KD, offset):
    return A * (x ** n) / (KD ** n + x ** n) + offset


rc('axes', linewidth=4)

experiment_paths = [r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_03_20_dpt-gfp_r4-gal4_ecoli-hs-dtom_input-output_pilot_4-6hrs']

df_name = 'bacteria.pkl'
method = 'bkg_sub_sum_data'

sigma_blur = 3
thresh = 360
dxy = 0.325

dpt_thresh = 700#240
n_ap_bins = 24#18
ap_bins = np.linspace(0, 1, n_ap_bins + 1)
plt.figure()

dpt_thresh_arr = 10 ** np.array([2.956, 2.968, 3.029, 2.8278])

available_volume = 2 * 0.325 * 0.325 * 500 ** 2 * np.array([  2, 116, 135, 167, 163, 150, 232, 255, 193, 201, 200, 185, 171,
       213, 194, 216, 251, 211, 185, 184])

x = np.linspace(0, 1, n_ap_bins)
xp = np.linspace(0, 1, len(available_volume))
sampled_volume = np.interp(x, xp, available_volume)

ap_min = 0#0.2
ap_max = 1#0.6

all_bacteria = []
all_dpt = []
all_ap = []
counter = 0
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

        ap_bin_volume = np.pi * (binned_larva_width / 2) ** 2 * ap_bin_width_microns
        #binned_concentration = binned_sum / ap_bin_volume * 10 ** 12
        binned_concentration = binned_sum / sampled_volume * 10 ** 12

        ap = ap_bins[:-1] / np.max(ap_bins[:-1])
        
        """compute dpt_line dist and bin"""
        dpt_mip = plt.imread(larvae_dir + '/mips_crop_gut/mip_crop_gut_channel0_t0.tif')
        dpt_line_dist = compute_dpt_line_dist_from_mip(dpt_mip, thresh=dpt_thresh_arr[counter])
        binned_dpt_line_dist = np.zeros(len(ap_bin_numbers))
        y_pixels = np.arange(mip.shape[0])
        
        tmp_df = pd.DataFrame({'dpt': dpt_line_dist, 'ap': y_pixels / np.max(y_pixels)})
        tmp_df = bin_aps(tmp_df, ap_bins, ap_col='ap')
        binned_dpt_line_dist = tmp_df.get(['dpt', 'binned_ap']).groupby(by='binned_ap').sum().values.flatten()
        # for i in range(len(ap_bin_numbers)):
        #     sel = np.abs(y_pixels / np.max(y_pixels) - ap_bins[ap_bin_numbers[i]]) == np.min(np.abs(y_pixels / np.max(y_pixels) - ap_bins[ap_bin_numbers[i]]))
        #     if np.sum(sel) > 1:
        #         sel[np.where(sel)[0][-1]] = False
        #     binned_dpt_line_dist[i] = dpt_line_dist[sel]
        
      
        sel = np.array(np.array(ap_bin_numbers) < (ap_max * n_ap_bins)) * np.array(np.array(ap_bin_numbers) > (ap_min * n_ap_bins))
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
        
        counter += 0

all_bacteria_arr = np.zeros((len(all_bacteria), len(all_bacteria[0])))
all_dpt_arr = np.zeros((len(all_dpt), len(all_dpt[0])))
all_ap_arr = np.zeros((len(all_ap), len(all_dpt[0])))

for i in range(len(all_bacteria_arr)):
    all_bacteria_arr[i] = all_bacteria[i]
    all_dpt_arr[i] = all_dpt[i]
    all_ap_arr[i] = all_ap[i]

all_bacteria_arr *= 1e-7
mean_bacteria = np.nanmean(all_bacteria_arr, axis=0)
mean_dpt = np.nanmean(all_dpt_arr, axis=0)
mean_ap = np.nanmean(all_ap_arr, axis=0)
std_bacteria = np.nanstd(all_bacteria_arr, axis=0)
std_dpt = np.nanstd(all_dpt_arr, axis=0)
std_ap = np.nanstd(all_ap_arr, axis=0)

# # plot double y of bacteria concentration and dpt-gfp
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.fill_between(mean_ap, mean_bacteria - std_bacteria, mean_bacteria + std_bacteria, color='m', linewidth=4, alpha=0.3)
# ax1.plot(mean_ap, mean_bacteria, 'r-', linewidth=4)
# ax2.fill_between(mean_ap, mean_dpt - std_dpt, mean_dpt + std_dpt, color='g', linewidth=4, alpha=0.3)
# ax2.plot(mean_ap, mean_dpt, 'g-', linewidth=4)
# ax1.tick_params(axis='y', labelcolor='m')
# ax1.set_label('bacterial concentration (10^7 cells/ml)')
# ax1.set_ylim([0, 75000])
# ax2.tick_params(axis='y', labelcolor='g')
# ax2.set_label('dpt-gfp (a.u.)')
# plt.xlim([np.min(mean_ap), np.max(mean_ap)])

# plot bacteria concentration and dpt-gfp
plt.figure()
plt.subplot(311)
plt.fill_between(mean_ap, mean_bacteria - std_bacteria, mean_bacteria + std_bacteria, color='m', linewidth=4, alpha=0.3)
plt.plot(mean_ap, mean_bacteria, 'm-', linewidth=4)
plt.xlim([np.min(mean_ap), np.max(mean_ap)])
plt.ylabel('planktonic bacteria \nconcentration \n($10^7$ cells/ml)', fontsize=24)
plt.xlabel('fraction of anterior-posterior axis', fontsize=24)
ax = plt.gca()
ax = style_axes(ax)
plt.tight_layout()

plt.subplot(312)
plt.fill_between(mean_ap, mean_dpt - std_dpt, mean_dpt + std_dpt, color='g', linewidth=4, alpha=0.3)
plt.plot(mean_ap, mean_dpt, 'g-', linewidth=4)
plt.ylabel('diptericin-gfp \nfluorescence \nintensity (a.u., 6 hpi)', fontsize=24)
plt.xlabel('fraction of anterior-posterior axis', fontsize=24)
plt.xlim([np.min(mean_ap), np.max(mean_ap)])
ax = plt.gca()
ax = style_axes(ax)
plt.tight_layout()

# plot 24 hour data
with open(r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/emL3_24_line_dist.pkl', 'rb') as f:
    line_dist_arr = pickle.load(f).values

m24 = np.mean(line_dist_arr, axis=0)
s24 = np.std(line_dist_arr, axis=0)
ap24 = np.linspace(0.15, 1, len(m24))
# binned_dpt_24 = np.zeros(len(mean_ap))
# binned_std_dpt_24 = np.zeros(len(mean_ap))
# for i in range(len(binned_dpt_24)):
#     sel = np.abs(ap24 - mean_ap[i]) == np.min(np.abs(ap24 - mean_ap[i]))
#     if np.sum(sel) > 1:
#         sel[np.where(sel)[0][-1]] = False
#     binned_dpt_24[i] = m24[sel]
#     binned_std_dpt_24[i] = s24[sel]
    
ap24 = np.linspace(0.18, 1, line_dist_arr.shape[1])
tmp_df = pd.DataFrame({'dpt': m24, 'std_dpt': s24, 'ap': ap24})
tmp_df = bin_aps(tmp_df, ap_bins, ap_col='ap')
binned_dpt_24 = tmp_df.get(['dpt', 'binned_ap']).groupby(by='binned_ap').mean().values.flatten()
binned_std_dpt_24 = tmp_df.get(['std_dpt', 'binned_ap']).groupby(by='binned_ap').mean().values.flatten()
ap_bins_24 = ap_bins[tmp_df.binned_ap.unique()]
sel = np.array(np.array(ap_bins_24) < ap_max) * np.array(np.array(ap_bins_24) > ap_min)
tmp_binned_dpt_24 = binned_dpt_24[sel]
tmp_binned_std_dpt_24 = binned_std_dpt_24[sel]
binned_dpt_24 = np.zeros_like(mean_dpt)
binned_std_dpt_24 = np.zeros_like(mean_dpt)
start_id = np.where(mean_ap > np.min(ap_bins_24))[0][0]
binned_dpt_24[:start_id] = 0
binned_std_dpt_24[:start_id] = 0
binned_dpt_24[start_id:] = tmp_binned_dpt_24
binned_std_dpt_24[start_id:] = tmp_binned_std_dpt_24
# for i in range(len(mean_dpt)):
#     if mean_ap[i] < np.min(ap_bins_24):
#         binned_dpt_24[i] = 0
#         binned_std_dpt_24[i] = 0
#     else:
#         binned_dpt_24[i] = tmp_binned_dpt_24[i]
#         binned_std_dpt_24[i] = tmp_binned_std_dpt_24[i]


plt.subplot(313)
#plt.fill_between(ap24, m24 - s24, m24 + s24, color='g', linewidth=4, alpha=0.3)
#plt.plot(ap24, m24, 'g-', linewidth=4)
plt.fill_between(mean_ap, binned_dpt_24 - binned_std_dpt_24, binned_dpt_24 + binned_std_dpt_24, color='g', linewidth=4, alpha=0.3)
plt.plot(mean_ap, binned_dpt_24, 'g-', linewidth=4)
ax = plt.gca()
ax = style_axes(ax)
plt.tight_layout()
plt.xlim([np.min(mean_ap), np.max(mean_ap)])
plt.ylabel('diptericin-gfp \nfluorescence \nintensity (a.u., 24 hpi)', fontsize=24)
plt.xlabel('fraction of anterior-posterior axis', fontsize=24)

"""plot individual dpt vs ap"""
plt.figure()
for i in range(len(all_dpt)):
    plt.plot(all_ap_arr[i], all_dpt_arr[i], linewidth=4, label=str(i))
plt.legend()
    
# first, fit exponential profiles to both 
# bacteria
x = mean_ap
y = mean_bacteria
exponential = lambda x, b, a, k: b + a * np.exp(-k * x)

p0 = (np.min(y), np.max(y) - np.min(y), np.log(np.max(y)/np.min(y)) / (np.max(x) - np.min(x)))
popt, pcov = curve_fit(exponential, x, y, p0=p0, bounds=((0, 0, 0), (np.max(y), np.max(y), np.inf)))

b, a, k = popt
bacteria_fit = exponential(x, b, a, k)

# dpt
x = mean_ap
y = mean_dpt
exponential = lambda x, b, a, k: b + a * np.exp(-k * x)

p0 = (90_000, np.max(y) - np.min(y), np.log(np.max(y)/np.min(y)) / (np.max(x) - np.min(x)))
popt, pcov = curve_fit(exponential, x, y, p0=p0, bounds=((90_000, 0, 0), (3 * np.max(y), 3 * np.max(y), np.inf)))

b, a, k = popt
dpt_fit = exponential(x, b, a, k)


"""plot input-output function"""
#sorted_ids = np.argsort(mean_bacteria)
sorted_ids = np.argsort(bacteria_fit)
bacteria_fit = bacteria_fit[sorted_ids]
mean_bacteria = mean_bacteria[sorted_ids]
mean_dpt = mean_dpt[sorted_ids]
std_bacteria = std_bacteria[sorted_ids]
std_dpt = std_dpt[sorted_ids]
mean_dpt_24 = binned_dpt_24[sorted_ids]
std_dpt_24 = binned_std_dpt_24[sorted_ids]


# fit hill function
p0 = (np.max(mean_dpt) - np.min(mean_dpt), 3, np.median(mean_bacteria), np.min(mean_dpt))    # A, n, KD, offset
popt, pcov = curve_fit(hill, bacteria_fit, mean_dpt, p0=p0, bounds=((0, 0, 0, 0), (np.max(mean_dpt), np.inf, np.max(mean_bacteria), np.max(mean_dpt))))
A, n, KD, offset = popt
x = np.linspace(0, 9, 1000)
y = hill(x, A, n, KD, offset)

# p0 = (np.max(mean_dpt) - np.min(mean_dpt), 3, np.median(mean_bacteria), np.min(mean_dpt))    # A, n, KD, offset
# popt, pcov = curve_fit(hill, mean_bacteria, mean_dpt, p0=p0, bounds=((0, 0, 0, 0), (np.max(mean_dpt), np.inf, np.max(mean_bacteria), np.max(mean_dpt))))
# A, n, KD, offset = popt
# x = np.linspace(0, 9, 1000)
# y = hill(x, A, n, KD, offset)

plt.figure()
#plt.plot(mean_bacteria, mean_dpt, 'ko', linewidth=4)
# plt.errorbar(mean_bacteria, mean_dpt, yerr=std_dpt, xerr=std_bacteria, 
#              ecolor='k', elinewidth=3, capsize=4, marker='o', 
#              markeredgewidth=2, markeredgecolor='k', markersize=18,
#              markerfacecolor='g', alpha=0.7, linewidth=4, color='none',
#              label='data')
plt.errorbar(bacteria_fit, mean_dpt, yerr=std_dpt, xerr=std_bacteria, 
             ecolor='k', elinewidth=3, capsize=4, marker='o', 
             markeredgewidth=2, markeredgecolor='k', markersize=18,
             markerfacecolor='g', alpha=0.7, linewidth=4, color='none',
             label='data')
plt.plot(x, y, '-', linewidth=6, color=np.array((75,0,130)) / 255, alpha=0.7,
         label='hill fit')

plt.xlabel('planktonic bacteria \nconcentration ($10^7$ cells/ml)', fontsize=24)
plt.ylabel('diptericin-gfp fluorescence \nintensity (a.u., 6 hpi)', fontsize=24)
plt.legend(fontsize=16)
#plt.xlim([1, 6])
ax = plt.gca()
ax = style_axes(ax)
plt.tight_layout()

# 24 
plt.figure()
#plt.plot(mean_bacteria, mean_dpt, 'ko', linewidth=4)
# plt.errorbar(mean_bacteria, mean_dpt_24, yerr=std_dpt_24, xerr=std_bacteria, 
#              ecolor='k', elinewidth=3, capsize=4, marker='o', 
#              markeredgewidth=2, markeredgecolor='k', markersize=18,
#              markerfacecolor='g', alpha=0.7, linewidth=4, color='none',
#              label='data')
plt.errorbar(bacteria_fit, mean_dpt_24, yerr=std_dpt_24, xerr=std_bacteria, 
             ecolor='k', elinewidth=3, capsize=4, marker='o', 
             markeredgewidth=2, markeredgecolor='k', markersize=18,
             markerfacecolor='g', alpha=0.7, linewidth=4, color='none',
             label='data')

# fit hill function
p0 = (np.max(mean_dpt_24) - np.min(mean_dpt_24), 3, np.median(mean_bacteria), np.min(mean_dpt_24))    # A, n, KD, offset
popt, pcov = curve_fit(hill, bacteria_fit, mean_dpt_24, p0=p0, bounds=((0, 0, 0, 0), (np.max(mean_dpt_24), np.inf, np.max(mean_bacteria), np.max(mean_dpt_24))))
A, n, KD, offset = popt
x = np.linspace(0, 9, 1000)
y = hill(x, A, n, KD, offset)
plt.plot(x, y, '-', linewidth=6, color=np.array((75,0,130)) / 255, alpha=0.7,
         label='hill fit')

plt.xlabel('planktonic bacteria \nconcentration ($10^7$ cells/ml)', fontsize=24)
plt.ylabel('diptericin-gfp fluorescence \nintensity (a.u., 24 hpi)', fontsize=24)
plt.legend(fontsize=16)
#plt.xlim([1, 6])
ax = plt.gca()
ax = style_axes(ax)
plt.tight_layout()


