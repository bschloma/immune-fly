from glob import glob
import pandas as pd
from quantify import quantify_nuclei
from segmentation import initialize_nuclei_dataframe


experiment_paths = [
    r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_06_ca-Gal4_UAS-His-RFP_mNG-EcR-B1']

quant_cols = ('ch0', 'ch1')

# 20x data
dxy = 0.325 * (2 ** 3)
dz = 2
voxel_size = (31, 14, 14)

seg_file_name = 'segmentation.ome.zarr/3'
im_file_name = 'im.ome.local_mean.zarr/3'
df_name = 'nuclei_3_quant_v4_bkg_by_slice.pkl'
for path in experiment_paths:
    larvae_dirs = glob(path + '/larva*')
    for larvae_dir in larvae_dirs:
        if larvae_dir == r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_06_ca-Gal4_UAS-His-RFP_mNG-EcR-B1/larva_1':
            continue
        path_to_seg = larvae_dir + '/' + seg_file_name
        path_to_im = larvae_dir + '/' + im_file_name
        print(larvae_dir)
        #if len(glob(larvae_dir + '/' + df_name)) > 0:
        #    continue
        df = initialize_nuclei_dataframe(path_to_seg, dxy=dxy, dz=dz, path_to_im=path_to_im, voxel_size=voxel_size)
        df = quantify_nuclei(df, quant_cols=quant_cols)

        df.to_pickle(larvae_dir + '/' + df_name)

