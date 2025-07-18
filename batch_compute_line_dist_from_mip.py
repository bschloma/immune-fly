import numpy as np
from skimage.io import imread
from quantify import compute_line_dist_from_mip
from pathlib import Path
from glob import glob
import pickle
import pandas as pd


experiment_paths = [r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2023_05_03-dpt-gfp_r4-gal4_uas-mcd8-mcherry_ecoli-hs-dtom_early-mid_24hrs_high_dose',
                    r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2023_06_01-dpt-gfp_r4-gal4_uas-mcd8-mcherry_ecoli-hs-dtom_early-mid_24hrs_high_dose',
                    r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2023_06_07-dpt-gfp_r4-gal4_uas-mcd8-mcherry_ecoli-hs-dtom_early-mid_24hrs_high_dose']

mip_dir = 'mips4'
mip_name = r'mip_green_0.tif'
save_name = 'line_dist_mean.pkl'
for experiment_path in experiment_paths:
    larvae_dirs = glob(experiment_path + '/larva*')
    for larvae_dir in larvae_dirs:
        print(larvae_dir)
        try:
            mip = imread(larvae_dir + '/' + mip_dir + '/' + mip_name)
        except FileNotFoundError as e:
            print(e)
            continue

        # y = np.linspace(0, mip.shape[0], 100)
        # x = int(mip.shape[1] / 2)
        # ap = pd.DataFrame({'x': x, 'y': y})
        line_dist = compute_line_dist_from_mip(mip)
        with open(larvae_dir + '/' + save_name, 'wb') as f:
            pickle.dump(line_dist, f)
