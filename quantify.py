import pandas as pd
import numpy as np
from skimage.morphology import binary_dilation, ball, disk
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm


def quantify_nuclei(df, quant_cols=('ch0',)):

    tqdm.pandas()
    for quant_col in quant_cols:
        res = df.progress_apply(quantify_nuclei_1row, axis=1, quant_col=quant_col).to_list()
        res0 = [l[0] for l in res]
        res1 = [l[1] for l in res]
        res2 = [l[2] for l in res]
        res_dict = {f'bkg_sub_mean_{quant_col}': res0, f'raw_mean_{quant_col}': res1, f'bkg_{quant_col}':res2}
        res_df = pd.DataFrame(res_dict)

        #res_df = pd.DataFrame(res, columns=[f'bkg_sub_mean_{quant_col}', f'raw_mean_{quant_col}', f'bkg_{quant_col}'])
        df = pd.concat((df.reset_index(), res_df), axis=1)

    return df


def quantify_nuclei_1row(row, quant_col):
    data = row.get(quant_col)
    seg = row.segments
    seg_id = row.seg_id
    #data, seg, seg_id = row
    seg = seg == seg_id

    """trying out background subtraction at each slice"""
    bkg_sub_mean = 0
    counter = 1
    for i in range(data.shape[0]):
        this_data = data[i]
        this_seg = seg[i]
        this_shell = binary_dilation(this_seg, disk(2)).astype('int') - seg.astype(int)
        bkg = np.mean(data[this_shell])

        if ~np.isnan(bkg):
            signal = this_data[this_seg]
            intens = signal - bkg
            intens[intens < 0] = 0
            this_mean = np.mean(intens)
            if ~np.isnan(this_mean):
                bkg_sub_mean += np.mean(intens)
                counter += 1

    bkg_sub_mean = bkg_sub_mean / counter

    signal = data[seg]

    shell = binary_dilation(seg, ball(2)).astype('int') - seg.astype(int)
    bkg = np.mean(data[shell])
    #bkg = np.median(data[shell])

    # intens = signal - bkg
    # intens[intens < 0] = 0
    # bkg_sub_mean = np.mean(intens)
    raw_mean = np.mean(signal)

    return bkg_sub_mean, raw_mean, bkg


def quantify_bacteria(df, quant_col='data'):
    tqdm.pandas()
    res = df.progress_apply(quantify_bacteria_1row, axis=1, quant_col=quant_col).to_list()
    res0 = [l[0] for l in res]
    res1 = [l[1] for l in res]
    res2 = [l[2] for l in res]
    res_dict = {f'bkg_sub_sum_{quant_col}': res0, f'raw_sum_{quant_col}': res1, f'bkg_{quant_col}':res2}
    res_df = pd.DataFrame(res_dict)

    #res_df = pd.DataFrame(res, columns=[f'bkg_sub_mean_{quant_col}', f'raw_sum_{quant_col}', f'bkg_{quant_col}'])
    df = pd.concat((df.reset_index(), res_df), axis=1)

    return df


def quantify_bacteria_1row(row, quant_col='data'):
    data = row.get(quant_col)
    """trying out background subtraction at each slice"""
    bkg_sub_sum = 0
    counter = 1
    all_bkgs = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        this_data = data[i]
        this_shell = binary_dilation(this_data > 0, disk(2)).astype('int') - (this_data > 0).astype(int)
        bkg = np.mean(this_data[this_shell])
        all_bkgs[i] = bkg
        if ~np.isnan(bkg):
            signal = this_data[this_data > 0]
            intens = signal - bkg
            intens[intens < 0] = 0
            this_mean = np.mean(intens)
            if ~np.isnan(this_mean):
                bkg_sub_sum += np.sum(intens)
                counter += 1

    bkg_sub_sum = bkg_sub_sum / counter

    signal = data[data > 0]

    #shell = binary_dilation(data > 0, ball(2)).astype('int') - (data > 0).astype(int)
    #bkg = np.mean(data[shell])
    bkg = np.mean(all_bkgs)
    #bkg = np.median(data[shell])

    raw_sum = np.sum(signal)

    return bkg_sub_sum, raw_sum, bkg
