import numpy as np
import pandas as pd
import zarr
from zarr.storage import DirectoryStore
from glob import glob
from pathlib import Path


experiment_paths = [r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_07_02-NP1029-Gal4_ecoli-hs-dtom_2hr']

zarr_paths = []
for path in experiment_paths:
    larvae_dirs = glob(path + '/larva*')
    for larvae_dir in larvae_dirs:
        zarr_paths.append(Path(larvae_dir) / 'bacteria.density.nmax2.ds50.ome.zarr/1')

for path in zarr_paths:
    im = np.squeeze(np.array(zarr.open(path, 'r')))
    available_pixels = np.sum(np.sum(im > 0, axis=0), axis=1)
    av_df = pd.DataFrame()
    av_df['available_pixels'] = available_pixels
    av_df.to_pickle(path.parent.parent / 'available_pixels.pkl')
