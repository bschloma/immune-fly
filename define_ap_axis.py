import napari
import pandas as pd
from skimage.io import imread
from scipy.interpolate import splprep, splev
import numpy as np
from pathlib import Path
from magicgui import magicgui
from napari.layers import Points


def get_df_from_points(points, save_dir=None):
    #points = viewer.layers[-1].data
    df = pd.DataFrame()
    df['y'] = points[:, 0]
    df['x'] = points[:, 1]

    if save_dir is not None:
        save_path = str(save_dir / 'manual_ap.pkl')
        df.to_pickle(save_path)

    return df


@magicgui(call_button='Interpolate')
def interpolate_points(points_layer: Points, n_bins: int=100) -> napari.layers.Points:
    points = points_layer.data
    # collect manually placed points
    df = get_df_from_points(points, save_dir=save_dir)
    points = df.values

    # create desired ap bins
    bins = np.linspace(0, 1, n_bins)

    # compute interpolated representation
    tck, u = splprep(points.T, k=1)

    # interpolate at bins
    interpolated_points = splev(bins, tck)

    # collect back into an array format that napari likes
    interpolated_points_array = np.zeros((n_bins, 2))
    for i in range(n_bins):
        interpolated_points_array[i, 0] = interpolated_points[0][i]
        interpolated_points_array[i, 1] = interpolated_points[1][i]

    # convert to a dataframe
    interpolated_points_df = pd.DataFrame(interpolated_points_array, columns=['y', 'x'])

    # optional save
    if save_dir is not None:
        save_path = str(save_dir / 'ap.pkl')
        interpolated_points_df.to_pickle(save_path)

    interpolated_points_layer = napari.layers.Points(interpolated_points_array, size=100, face_color='cyan')

    return interpolated_points_layer

# set up
path_to_mip = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2023_05_03-dpt-gfp_r4-gal4_uas-mcd8-mcherry_ecoli-hs-dtom_early-mid_24hrs_high_dose/larvae_6/mips/mip_red_0.tif'
save_dir = Path(path_to_mip).parent.parent / 'ap'
save_dir.mkdir(exist_ok=True)
mip = imread(path_to_mip)

# launch the viewer and set up the widget
viewer = napari.view_image(mip, colormap='magenta')
viewer.layers[0].contrast_limits = [0, 20_000]
dock = viewer.window.add_dock_widget(interpolate_points, area='right')


