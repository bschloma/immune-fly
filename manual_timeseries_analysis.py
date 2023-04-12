"""program to draw points on a mip timeseries to use for manually quantifying dpt levels in cells."""
import napari
from PIL import Image
import numpy as np
from glob import glob
import pandas as pd
from pathlib import Path


def get_df_from_points(timeseries=False, save_path=None):
    points = viewer.layers[-1].data
    df = pd.DataFrame()
    if timeseries:
        points['t'] = points[:, 0]
    df['y'] = points[:, 1]
    df['x'] = points[:, 2]

    if save_path is not None:
        df.to_pickle(save_path)

    return df


# load the mips into a numpy array
path_to_mips = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_4/mips'
save_path = Path(path_to_mips).parent / 'manual_df.pkl'

# load green mips
prefix = r'mip_green_'
suffix = r'.tif'
files = glob(path_to_mips + '/' + prefix + '*.tif')
num_ims = len(files)

for i in range(num_ims):
    this_filename = path_to_mips + '/' + prefix + str(i) + suffix
    this_im = np.array(Image.open(this_filename))
    if i == 0:
        im_arr_green = np.zeros((num_ims, this_im.shape[0], this_im.shape[1]))
    im_arr_green[i] = this_im

# load red mips
prefix = r'mip_red_'
suffix = r'.tif'
files = glob(path_to_mips + '/' + prefix + '*.tif')
num_ims = len(files)

for i in range(num_ims):
    this_filename = path_to_mips + '/' + prefix + str(i) + suffix
    this_im = np.array(Image.open(this_filename))
    if i == 0:
        im_arr_red = np.zeros((num_ims, this_im.shape[0], this_im.shape[1]))
    im_arr_red[i] = this_im

viewer = napari.view_image(im_arr_green)
viewer.add_image(im_arr_red)

