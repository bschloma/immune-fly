#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:21:25 2024

@author: brandon
"""

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
from glob import glob
from scipy.ndimage import gaussian_filter
import matplotlib as mpl


mpl.rc('axes', linewidth=4)
fontsize=24
mpl.rc('font', family='Arial')

def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    
    plt.tight_layout()
    
    return ax


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


#experiment_paths = [r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_03_20_dpt-gfp_r4-gal4_ecoli-hs-dtom_input-output_pilot_4-6hrs']
# note: recent 5x datasets are flipped due to mounting upside down. 
# experiment_paths = [
#     r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_06_04-NP1029-Gal4_UAS-Mhc-RNAi_ecoli-hs-dtom_4hrs/no_heartbeat']
# experiment_paths = [
#     r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_06_18-NP1029-Gal4_ecoli-hs-dtom_1hr']
#experiment_paths = [
#    r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_05_30-dpt-gfp_r4-gal4_ecoli-hs-dtom_4hrs_flow_field']
experiment_paths = [r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_06_04-NP1029-Gal4_UAS-Mhc-RNAi_ecoli-hs-dtom_4hrs/no_heartbeat']#, 
                    #r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_06_26-NP1029-Gal4_UAS-Mhc-RNAi_ecoli-hs-dtom_4hrs/no_heartbeat']
#experiment_paths = [r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_07_02-NP1029-Gal4_ecoli-hs-dtom_2hr']
df_name = 'bacteria.pkl'
method = 'bkg_sub_sum_data'

sigma_blur = 3
thresh = 220
dxy = 0.325



n_ap_bins = 14#22
ap_bins = np.linspace(0, 1, n_ap_bins + 1)

#available_volume = 2 * 0.325 * 0.325 * 500 ** 2 * np.array([ 116, 135, 167, 163, 150, 232, 255, 193, 201, 200, 185, 171,
#       213, 194, 216, 251, 211, 185, 184])

# available_volume = 4 * 0.91 * 0.91 * 50 ** 2 * np.array([  52.6,  126.4,  237.4,  346.8,  508.2,  751.8,  927.6,  981. ,
#        1226.2,  930.6, 1203.2,  874.2,  925.6, 1110.8,  961.2, 1028.2,
#        1055.6, 1155. , 1144.6,  951.4,  672. ,  191.6])

#available_volume = 4 * 0.91 * 0.91 * 20 ** 2 * np.array([ 54.  , 119.  , 186.5 , 240.5 , 254.5 , 259.  , 281.  , 289.75,
#       281.25, 306.75, 281.75, 299.25, 296.25, 273.25, 283.  , 261.75,
#       244.5 , 226.5 , 213.5 , 169.  , 126.75,  35.  ])

available_volume = np.array([3.49238605e+07, 6.53988130e+07, 1.01998991e+08, 1.25074541e+08,
       1.46105305e+08, 1.56498386e+08, 1.62622979e+08, 1.80517528e+08,
       1.77813337e+08, 1.90367745e+08, 1.81240248e+08, 1.79402863e+08,
       1.75391286e+08, 1.69207341e+08, 1.61747398e+08, 1.47125100e+08,
       1.39735655e+08, 1.09845337e+08, 9.37926961e+07, 6.89674743e+07,
       4.08586640e+07, 1.99802229e+07])

x = np.linspace(0, 1, n_ap_bins)
xp = np.linspace(0, 1, len(available_volume))
sampled_volume = np.interp(x, xp, available_volume)
sampled_volume = np.flip(sampled_volume)

all_ap_dists = np.zeros((6, n_ap_bins))
all_ap_dens = np.zeros((6, n_ap_bins))

fb_shift = 0.12

plt.figure()
counter = 0
for path in experiment_paths:
    larvae_dirs = glob(path + '/larva*')
    for larvae_dir in larvae_dirs:
        if larvae_dir == r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_03_20_dpt-gfp_r4-gal4_ecoli-hs-dtom_input-output_pilot_4-6hrs/larva_1':
            continue
        
        if larvae_dir == r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_06_04-NP1029-Gal4_UAS-Mhc-RNAi_ecoli-hs-dtom_4hrs/no_heartbeat/larva_1':
            continue
        
        if larvae_dir == r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_06_04-NP1029-Gal4_UAS-Mhc-RNAi_ecoli-hs-dtom_4hrs/no_heartbeat/larva_3':
            continue
        
        if larvae_dir == r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_07_02-NP1029-Gal4_ecoli-hs-dtom_2hr/larva_2':
            continue
        
        if larvae_dir == r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_07_02-NP1029-Gal4_ecoli-hs-dtom_2hr/larva_3':
            continue
        
        if larvae_dir == r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_06_26-NP1029-Gal4_UAS-Mhc-RNAi_ecoli-hs-dtom_4hrs/no_heartbeat/larva_5':
            continue
        
        if len(glob(larvae_dir + '/' + df_name)) == 0:
            continue
        df = pd.read_pickle(larvae_dir + '/' + df_name)
        func = lambda row: row.data[0].size
        sizes = df.apply(func, axis=1)
        df = df[sizes < 1e5]

        df['ap'] = (df.y.max() - df.y) / df.y.max()
        #df['ap'] = df.y / df.y.max()

        df = bin_aps(df, ap_bins, ap_col='ap')
        ap_col = 'ap'
        ap_bin_width_microns = df.y.max() * dxy / n_ap_bins

        intens = df.get(method)
        single_cell_inten = np.median(intens)
        #df['n_bacteria'] = intens / single_cell_inten

        df['n_bacteria'] = np.clip(intens / single_cell_inten, a_min=1, a_max=np.inf).astype('int')
        
        #df = df[df.n_bacteria < 3]
        """plot number of bacteria vs ap axis"""
        binned_sum = df.get(['n_bacteria', f'binned_{ap_col}']).groupby(by=f'binned_{ap_col}').sum().values.flatten()
        ap_bin_numbers = sorted(df.get(f'binned_{ap_col}').unique())

        #binned_sum = binned_sum / np.max(binned_sum)
        # normalize by larva cross sectional area
        # mip = plt.imread(larvae_dir + '/mips/mip_red_0.tif')
        # larva_width = compute_larva_width_from_mip(mip, sigma_blur, thresh, dxy=dxy)
        
        # binned_larva_width = np.zeros(len(ap_bin_numbers))
        # y_pixels = np.arange(mip.shape[0])
        # for i in range(len(ap_bin_numbers)):
        #     sel = np.abs(y_pixels / np.max(y_pixels) - ap_bins[ap_bin_numbers[i]]) == np.min(np.abs(y_pixels / np.max(y_pixels) - ap_bins[ap_bin_numbers[i]]))
        #     if np.sum(sel) > 1:
        #         sel[np.where(sel)[0][-1]] = False
        #     binned_larva_width[i] = larva_width[sel]
        
        # ap_bin_volume = (4 * np.pi / 3) * (binned_larva_width / 2) ** 2 * ap_bin_width_microns
        #binned_concentration = binned_sum# / ap_bin_volume * 10 ** 9
        binned_concentration = binned_sum / sampled_volume * 1e12 / 1e7# / ap_bin_volume * 10 ** 9
        all_ap_dists[counter, ap_bin_numbers] = binned_sum
        all_ap_dens[counter, ap_bin_numbers] = binned_concentration

        ap = (ap_bins[:-1] / np.max(ap_bins[:-1]) - fb_shift) / (1 - fb_shift)
        plt.plot(ap[ap_bin_numbers], binned_sum, '-', linewidth=4, label=str(counter))
        #plt.plot(ap, binned_sum, '-', linewidth=4)

        plt.xlabel('fraction of anterior-posterior axis', fontsize=24)
        #plt.ylabel('bacterial density (cells/ml^3', fontsize=24)
        plt.ylabel('bacteria counts (num. cells)', fontsize=24)
        plt.legend()
        counter += 1
        #plt.xlim([0.1, 1.0])
        #plt.ylim([0, 5e-4])
        

plt.figure()
m = np.mean(all_ap_dists, axis=0)
s = np.std(all_ap_dists, axis=0)
u = m + s
l = m - s
plt.fill_between(ap, l, u, color='m', alpha=0.5)
plt.plot(ap, m, 'm-', linewidth=4)
plt.xlabel('fraction of anterior-posterior axis', fontsize=24)
plt.ylabel('bacteria counts \n(num. cells)', fontsize=24)
ax = style_axes(plt.gca())


plt.figure(figsize=(8,4))
m = np.mean(all_ap_dens, axis=0)
s = np.std(all_ap_dens, axis=0) #/ np.sqrt(len(all_ap_dens))
u = m + s
l = m - s
plt.fill_between(ap, l, u, color='m', alpha=0.5)
plt.plot(ap, m, 'm-', linewidth=4)
plt.xlabel('fraction of anterior-posterior axis', fontsize=24)
plt.ylabel('bacteria density \n($10^7$ cells / ml)', fontsize=24)
ax = style_axes(plt.gca())
#plt.xlim([0, 1])
#plt.ylim([0, 0.4E9])


