import numpy as np
import zarr
from dexp.datasets import ZDataset
import pandas as pd


path_to_culled_labels_zarr = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_1/ds_h5/scan_0/segmentation_mip_culled.zarr'
path_to_ds = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_1/crop_chk.zarr'
ds = ZDataset(path_to_ds, mode='r')

dpt_mip = np.array(ds.get_projection_array('dpt-red', axis=0))
seg = zarr.open(path_to_culled_labels_zarr)

ids = np.unique(seg)
ids = ids[ids > 0]

#df = pd.DataFrame(columns='mean_amp_level')
mean_amp_levels = np.zeros((len(ids), 1))
for i in range(len(ids)):
    print(str(i) + ', of ' + str(len(ids)))
    these_pixels = dpt_mip[np.array(seg) == ids[i]]
    mean_amp_levels[i] = np.mean(these_pixels)

#df['mean_amp_level'] = mean_amp_levels

