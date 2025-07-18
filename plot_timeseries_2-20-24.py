#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 13:31:16 2024

@author: brandon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from glob import glob
from matplotlib import rc
from matplotlib.colors import ListedColormap


#df = pd.read_pickle(r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_20_dpt-gfp_ca-Gal4_UAS-His-RFP_halocarbon_5x_timeseries/larva_2/df_quant.pkl')
df = pd.read_pickle(r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_14_dpt-gfp_ca-Gal4_UAS-His-RFP_halocarbon_5x_timeseries/larva_2/manual_analysis/all_cells_quant.pkl')

nuc_ids = df.particle.unique()
for nid in nuc_ids:
    sub_df = df[df.particle == nid]
    sub_df = sub_df.sort_values(by='t')
    t = sub_df.t
    dpt = sub_df.mean_dpt
    
    plt.figure()
    plt.plot(sub_df.t * 2, sub_df.mean_dpt, linewidth=3)
    plt.xlabel('time (min)')
    plt.ylabel('mean dpt (a.u.)')
    plt.xlim([0, 90])
    plt.ylim([200, 1100])
    

df1 = df[df.y < 1000]
df2 = df[np.array(df.y.values >= 1000) * np.array(df.y.values < 2000)]
df3 = df[df.y > 2000]

# anterior
plt.figure()
plt.subplot(2,1,1)
nuc_ids = df1.particle.unique()
for nid in nuc_ids:
    sub_df = df1[df1.particle == nid]
    sub_df = sub_df.sort_values(by='t')
    t = sub_df.t
    dpt = sub_df.mean_dpt
    
    plt.plot(sub_df.t * 2, sub_df.mean_dpt, linewidth=3)
    plt.xlabel('time (min)')
    plt.ylabel('mean dpt (a.u.)')
    plt.xlim([0, 90])
    plt.ylim([200, 1300])
    #plt.xlim([0, 90])
plt.title('anterior')
    
# middle
plt.subplot(2,1,2) 
nuc_ids = df2.particle.unique()
for nid in nuc_ids:
    sub_df = df2[df2.particle == nid]
    sub_df = sub_df.sort_values(by='t')
    t = sub_df.t
    dpt = sub_df.mean_dpt
    
    plt.plot(sub_df.t * 2, sub_df.mean_dpt, linewidth=3)
    plt.xlabel('time (min)')
    plt.ylabel('mean dpt (a.u.)')
    plt.xlim([0, 180])
    plt.ylim([200, 1300])
    #plt.xlim([0, 90])
plt.title('middle')
    
# # middle
# plt.subplot(3,1,3) 
# nuc_ids = df3.particle.unique()
# for nid in nuc_ids:
#     sub_df = df3[df3.particle == nid]
#     sub_df = sub_df.sort_values(by='t')
#     t = sub_df.t
#     dpt = sub_df.mean_dpt
    
#     plt.plot(sub_df.t * 2, sub_df.mean_dpt, linewidth=3)
#     plt.xlabel('time (min)')
#     plt.ylabel('mean dpt (a.u.)')
#     plt.xlim([0, 180])
#     plt.ylim([200, 1100])
#     #plt.xlim([0, 90])
# plt.title('posterior')
"plot by y coordinate"
# def get_y_bin(y, y_bins):
#     distances = np.abs(y - y_bins)
#     this_y_bin = y_bins[distances == np.min(distances)]
    
#     return this_y_bin
    

# def get_ap_for_spots(df, ap):
#     """df = spots df, ap = ap df"""
#     y_bins = np.zeros(len(df))
#     ys = df.y.values.tolist()
#     for i in range(len(ys)):
#         y_bins[i] = get_y_bin(locs[i], ap)
    
#     df['y_bin'] = y_bins
    
#     return df


nuc_ids = df2.particle.unique()
slope_arr = np.zeros(len(nuc_ids))
mean_y_arr = np.zeros(len(nuc_ids))
first_y_arr = np.zeros(len(nuc_ids))
for i in range(len(nuc_ids)):
    sub_df = df[df.particle == nuc_ids[i]]
    sub_df = sub_df.sort_values(by='t')
    t = sub_df.t
    dpt = sub_df.mean_dpt
    
    dpt = dpt[t <= 40]
    t = t[t <= 40]
    
    dpt = dpt[t > 10]
    t = t[t > 10]
    
    slope, intercept, r, p, se = linregress(2 * t, dpt)
    slope_arr[i] = slope
    
    mean_y_arr[i] = sub_df.y.mean()
    try:
        first_y_arr[i] = sub_df[sub_df.t == sub_df.t.min()].y
    except ValueError:
        first_y_arr[i] = np.nan

plt.figure()
#plt.plot(mean_y_arr, slope_arr, 'ko')
plt.plot(first_y_arr, slope_arr, 'ko')
plt.xlabel('ap position (px)')
plt.ylabel('activation rate (a.u./min)')
    
    
"""plot individual traces colored by slope"""
slope_bins = np.linspace(0, 4, 100)
reds = np.linspace(1, 0, len(slope_bins))
greens = np.linspace(0, 1, len(slope_bins))
blues = np.ones(len(slope_bins))

fontsize = 24
fontweight = 'bold'
fontproperties = {'family':'sans-serif','sans-serif':['Arial'],'weight' : fontweight, 'size' : fontsize}
rc('axes', linewidth=4)

t_fit_max = 12
t_fit_min = 0

plt.figure()

# anterior
plt.subplot(121)
this_df = df1
nuc_ids = this_df.particle.unique()
slope_arr = np.zeros(len(nuc_ids))
mean_y_arr = np.zeros(len(nuc_ids))
for i in range(len(nuc_ids)):
    sub_df = this_df[this_df.particle == nuc_ids[i]]
    sub_df = sub_df.sort_values(by='t')
    if any(sub_df.groupby(by='t').size().values > 1):
        continue
    t = sub_df.t
    dpt = sub_df.mean_dpt
    
    dpt = dpt[t <= t_fit_max]
    t = t[t <= t_fit_max]
    
    dpt = dpt[t > t_fit_min]
    t = t[t > t_fit_min]
    
    if len(t) > 4:
        slope, intercept, r, p, se = linregress(2 * t, dpt)
        
        slope_index = np.where(np.abs(slope - slope_bins) == np.min(np.abs(slope - slope_bins)))[0][0]
        
        t = sub_df.t
        dpt = sub_df.mean_dpt
        plt.plot((t * 2 / 60) + 5, dpt, linewidth=2, color=[reds[slope_index], greens[slope_index], blues[slope_index]])

plt.xlabel('hours post infection', fontsize=fontsize)
plt.ylabel('mean diptericin-GFP \nintensity per cell (a.u.)', fontsize=fontsize)
plt.title('anterior', fontsize=fontsize)
plt.ylim([200, 1300])
plt.xlim([5, 8])
ax = plt.gca()
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)

plt.tight_layout()

# posterior
plt.subplot(122)
this_df = df2
nuc_ids = this_df.particle.unique()
slope_arr = np.zeros(len(nuc_ids))
mean_y_arr = np.zeros(len(nuc_ids))
for i in range(len(nuc_ids)):
    sub_df = this_df[this_df.particle == nuc_ids[i]]
    sub_df = sub_df.sort_values(by='t')
    if any(sub_df.groupby(by='t').size().values > 1):
        continue
    t = sub_df.t
    dpt = sub_df.mean_dpt
    
    dpt = dpt[t <= t_fit_max]
    t = t[t <= t_fit_max]
    
    dpt = dpt[t > t_fit_min]
    t = t[t > t_fit_min]
    
    if len(t) > 4:
        slope, intercept, r, p, se = linregress(2 * t, dpt)
        
        slope_index = np.where(np.abs(slope - slope_bins) == np.min(np.abs(slope - slope_bins)))[0][0]
        
        t = sub_df.t
        dpt = sub_df.mean_dpt
        plt.plot(t * 2 / 60 + 5, dpt, linewidth=2, color=[reds[slope_index], greens[slope_index], blues[slope_index]])

plt.xlabel('hours post infection', fontsize=fontsize)
plt.ylabel('mean diptericin-GFP \nintensity per cell (a.u.)', fontsize=fontsize)
plt.title('middle', fontsize=fontsize)
plt.ylim([200, 1300])
plt.xlim([5, 8])
ax = plt.gca()
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)

plt.tight_layout()

"""colorbar"""
N = len(slope_bins)
vals = np.ones((N, 4))
vals[:, 0] = reds
vals[:, 1] = greens
vals[:, 2] = blues
newcmp = ListedColormap(vals)
tmp_im = np.array([slope_bins, slope_bins])

fig, ax = plt.subplots()
cax = ax.imshow(tmp_im, cmap=newcmp)


# Add colorbar, make sure to specify tick locations to match desired ticklabels
cbar = fig.colorbar(cax, ticks=[0, 2, 4])
cbar.ax.set_yticklabels([0, 2, 4], fontsize=24)  # vertically oriented colorbar


# plt.figure()

# #plt.imshow(tmp_im, cmap=newcmp)
# plt.colorbar()