import pandas as pd
from segmentation import quantify
import numpy as np


# df = pd.read_pickle(r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_20_dpt-gfp_ca-Gal4_UAS-His-RFP_halocarbon_5x_timeseries/larva_2/tracks_culled.pkl')
# path_to_particles =  r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_20_dpt-gfp_ca-Gal4_UAS-His-RFP_halocarbon_5x_timeseries/larva_2/segmentation.tracked.zarr'
# path_to_im_zarr = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_20_dpt-gfp_ca-Gal4_UAS-His-RFP_halocarbon_5x_timeseries/larva_2/im.ome.zarr/0'
# channel = 0
# fun = np.mean
# fun_name = 'mean_dpt'

df = pd.read_pickle(r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_06_ca-Gal4_UAS-His-RFP_mNG-EcR-B1/larva_2/nuclei.pkl')
df['particle'] = df['seg_id']
path_to_particles =  r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_06_ca-Gal4_UAS-His-RFP_mNG-EcR-B1/larva_2/segmentation.ome.zarr/4'
path_to_im_zarr = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_06_ca-Gal4_UAS-His-RFP_mNG-EcR-B1/larva_2/im.ome.local_mean.zarr/4'
channel = 0
fun = np.mean
fun_name = 'mean_dpt'


df = quantify(df, path_to_particles, path_to_im_zarr, channel, fun, fun_name)