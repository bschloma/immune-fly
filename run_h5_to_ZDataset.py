import numpy as np
from readwrite import individual_hdf5s_to_ZDataset
import napari
from dexp.datasets import ZDataset

path_to_h5_dir = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ctl_no_ether_no_inj/larvae_1/ds_h5/scan_0/PreProcessing/confocal_2D_unet_bce_dice_ds3x'
path_to_new_zarr = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ctl_no_ether_no_inj/larvae_1/ds_h5/scan_0/predictions.zarr'
z_slices = None
prefix = 'z_'
suffix = '_predictions.h5'
dtype = np.float32

individual_hdf5s_to_ZDataset(path_to_h5_dir, path_to_new_zarr, z_slices, prefix, suffix, dtype)

#ds = ZDataset(path_to_new_zarr, 'r')
#viewer = napari.view_labels(ds.get_array("predictions"))
#viewer = napari.view_image(ds.get_array("predictions"))

