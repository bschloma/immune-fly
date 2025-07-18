import numpy as np
import pandas as pd
from skimage.filters import gaussian, threshold_otsu
from skimage.transform import downscale_local_mean
from skimage.morphology import binary_closing, disk
from skimage.io import imread
from scipy.ndimage import binary_fill_holes
import napari
from glob import glob
from pathlib import Path


experiment_paths = [r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_07_02-NP1029-Gal4_ecoli-hs-dtom_2hr']
mip_paths = []
for path in experiment_paths:
    larvae_dirs = glob(path + '/larva*')
    for larvae_dir in larvae_dirs:
        mip_paths.append(Path(larvae_dir) / 'mips/mip_green_0.tif')

thresh = 220
for path in mip_paths:
    im = imread(path)
    im_ds = downscale_local_mean(im, 20)
    mask = im_ds > thresh
    mask = binary_closing(mask, disk(3))
    mask = binary_fill_holes(mask)
    width = np.sum(mask, axis=1)
    width_df = pd.DataFrame()
    width_df['width'] = width
    width_df.to_pickle(path.parent.parent / 'width.pkl')
