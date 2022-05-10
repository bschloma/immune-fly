import numpy as np
from readwrite import ZDataset_to_individual_hdf5s


path_to_ds = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ctl_no_ether_no_inj/larvae_1/crop.zarr'
path_to_new_hdf5_dir = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ctl_no_ether_no_inj/larvae_1/ds_h5'
channels = ["mem-green"]
timepoints = None
z_slices = None

ZDataset_to_individual_hdf5s(path_to_ds, path_to_new_hdf5_dir, channels, timepoints, z_slices)
