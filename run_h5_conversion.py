import numpy as np
from readwrite import ZDataset_to_individual_hdf5s, ome_zarr_to_individual_hdf5s


path_to_ds = r'/media/brandon/Data1/Brandon/fly_immune/Serenity/2022_08_24_dpt-gfp_silverman_r4-gal4_uas-mcd8-mcherry_ecoli-hs-dtom/larvae_1/crop.ome.zarr'
path_to_new_hdf5_dir = r'/media/brandon/Data1/Brandon/fly_immune/Serenity/2022_08_24_dpt-gfp_silverman_r4-gal4_uas-mcd8-mcherry_ecoli-hs-dtom/larvae_1/ds_h5'
channels = ["mem-red"]#
channel_numbers = [1]
channel_names = ['mem-red']
timepoints = None
z_slices = None

#ZDataset_to_individual_hdf5s(path_to_ds, path_to_new_hdf5_dir, channels, timepoints, z_slices)
ome_zarr_to_individual_hdf5s(path_to_ds, path_to_new_hdf5_dir, channel_numbers, channel_names, timepoints, z_slices)
