from glob import glob
import pandas as pd
from quantify import quantify_bacteria
from segmentation import initialize_bacteria_dataframe


experiment_paths = [
    r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_07_02-NP1029-Gal4_ecoli-hs-dtom_2hr']

quant_col = 'data'

# 20x data
# dxy = 0.325
# dz = 2
# channel = 1

# 5x data
dxy = 0.91
dz = 4
channel = 0

seg_file_name = 'bacteria.segmentation.ome.zarr/0'
im_file_name = 'im.ome.zarr/0'
df_name = 'bacteria.pkl'
for path in experiment_paths:
    larvae_dirs = glob(path + '/larva*')
    for larvae_dir in larvae_dirs:
        #if larvae_dir == r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_06_04-NP1029-Gal4_UAS-Mhc-RNAi_ecoli-hs-dtom_4hrs/no_heartbeat/larva_1':
        #    continue
        path_to_seg = larvae_dir + '/' + seg_file_name
        if len(glob(path_to_seg)) == 0:
            continue
        path_to_im = larvae_dir + '/' + im_file_name
        print(larvae_dir)
        if len(glob(larvae_dir + '/' + df_name)) > 0:
            continue
        print('initializing dataframe')
        df = initialize_bacteria_dataframe(path_to_seg, path_to_im=path_to_im, dxy=dxy, dz=dz, channel=channel)
        print('quantifying bacteria')
        df = quantify_bacteria(df, quant_col=quant_col)

        df.to_pickle(larvae_dir + '/' + df_name)

