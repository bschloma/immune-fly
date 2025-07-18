from segmentation import create_bacteria_density_map
import pandas as pd
import numpy as np


larva_dir = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_06_26-NP1029-Gal4_UAS-Mhc-RNAi_ecoli-hs-dtom_4hrs/no_heartbeat/larva_1'
df = pd.read_pickle(larva_dir + r'/bacteria.pkl')
intens = df.get('bkg_sub_sum_data')
single_cell_inten = np.median(intens)
df['n_bacteria'] = np.clip(intens / single_cell_inten, a_min=1, a_max=np.inf).astype('int')
df = df[df.n_bacteria < 3]

path_to_image_zarr = larva_dir + r'/im.ome.zarr/0'
path_to_density_ome_zarr = larva_dir + r'/bacteria.density.nmax2.ds200.ome.zarr'

create_bacteria_density_map(df, path_to_density_ome_zarr=path_to_density_ome_zarr, path_to_image_zarr=path_to_image_zarr, method='n_bacteria', pyramid_scales=2, downscale=200)
