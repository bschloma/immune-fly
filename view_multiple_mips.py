import napari
from glob import glob
from skimage.io import imread
import numpy as np


experiment_paths = [
    r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_04_18_dpt-gfp_NP1029-Gal4-UAS-Mhc-RNAi_ecoli-hs-dtom_6hrs/yes_heartbeat',
    r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_04_19_dpt-gfp_NP1029-Gal4-UAS-Mhc-RNAi_ecoli-hs-dtom_6hrs/yes_heartbeat']


mip_dir_name = 'mips4'
viewer = napari.Viewer()
for path in experiment_paths:
    larvae_dirs = glob(path + '/larva*')
    for larvae_dir in larvae_dirs:
        mip_files = glob(larvae_dir + f'/{mip_dir_name}/*.tif')
        for i, mip_file in enumerate(mip_files):
            this_mip = imread(mip_file)
            if i == 0:
                mip_arr = np.zeros((len(mip_files), this_mip.shape[0], this_mip.shape[1]))
            mip_arr[i] = this_mip

        viewer.add_image(mip_arr, channel_axis=0)




