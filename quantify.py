import pandas as pd
import numpy as np
from skimage.morphology import binary_dilation, ball, disk
from skimage.filters import gaussian
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from skimage.filters import threshold_multiotsu


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


def compute_line_dist_from_mip(mip, bkg=None, ap=None, method='mean'):
    # if background is not supplied, calculate it using a multiotsu method
    if bkg is None:
        bkg = 10 ** threshold_multiotsu(np.log10(mip[mip > 0]))[-1]

    # subtract background
    mip = mip.astype('float32')
    mip -= bkg
    mip[mip < 0] = 0

    if ap is None:
        short_axis = np.where(mip.shape == np.min(mip.shape))[0][0]
        #line_dist = np.sum(mip, axis=short_axis)
        if method == 'mean':
            line_dist = np.sum(mip, axis=short_axis) / np.sum(mip > 0, axis=short_axis)
        elif method == 'sum':
            line_dist = np.sum(mip, axis=short_axis)
        else:
            raise ValueError('compute_line_dist_from_mip: method can only be mean or sum')

    else:
        intens = mip[mip > 0]
        Y, X = np.indices(mip.shape)
        good_Y = Y[mip > 0]
        good_X = X[mip > 0]

        line_dist = np.zeros(len(ap))

        for i in range(len(intens)):
            loc = (good_Y[i], good_X[i])
            inten = intens[i]
            this_ap = get_ap(loc, ap)
            line_dist[int(this_ap)] += inten

    return line_dist


def get_ap(loc, ap):
    """loc = (y,x) of interest, ap = ap DF"""
    ap_vals = ap.get(['y', 'x']).values
    distances = distance(loc, ap_vals)
    ap_value = np.argwhere(distances == np.nanmin(distances))
    if len(ap_value) > 1:
        ap_value = ap_value[0]

    return ap_value


def distance(x, y):
    """ x = 1xd, y = Nxd"""
    return np.sqrt(np.sum((x - y) ** 2, axis=1))
