"""program to draw points on a mip timeseries to use for manually quantifying dpt levels in cells."""
import napari
from PIL import Image
import numpy as np
from glob import glob
import pandas as pd
from pathlib import Path


def get_df_from_points(timeseries=False, save_path=None, cell=None):
    points = viewer.layers[-1].data
    df = pd.DataFrame()
    if timeseries:
        df['t'] = points[:, 0]
    df['y'] = points[:, 1]
    df['x'] = points[:, 2]

    if save_path is not None:
        df.to_pickle(save_path / f'cell_{cell}.pkl')

    return df


# load the mips into a numpy array
path_to_mips = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_14_dpt-gfp_ca-Gal4_UAS-His-RFP_halocarbon_5x_timeseries/larva_2/mips'
save_path = Path(path_to_mips).parent / 'manual_analysis'
cell = 39

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

viewer = napari.view_image(im_arr_green, colormap='green', blending='additive', contrast_limits=(0, 30_000))
viewer.add_image(im_arr_red, colormap='magenta', blending='additive', contrast_limits=(0, 20_000))

# pred = np.array(Image.open(Path(path_to_mips).parent / 'prediction_mip.tif'))
# viewer.add_image(pred, colormap='blue', blending='additive', contrast_limits=(0, 0.5))

