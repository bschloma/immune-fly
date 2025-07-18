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
    df['z'] = points[:, 0]
    df['y'] = points[:, 1]
    df['x'] = points[:, 2]

    if save_path is not None:
        df.to_pickle(save_path)

    return df


# load the mips into a numpy array
path_to_ome_zarr = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2023_07_11-mNG-EcR_early_mid/larvae_3/im.ome.zarr'
save_path = Path(path_to_ome_zarr).parent / 'manual_df.pkl'
channel_axis = 1

viewer = napari.Viewer()
viewer.open(path_to_ome_zarr, channel_axis=channel_axis)

