from segmentation import segment_bacteria
import zarr
from zarr.storage import DirectoryStore
import napari
import numpy as np
import dask.array as da
from glob import glob

experiment_paths = [
    r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_07_02-NP1029-Gal4_ecoli-hs-dtom_2hr']

# sigma_blur = 1.0
# bright_thresh = 0.04
# sigma_low = 1.0
# sigma_high = 2.0
# disk_size = 3
# bacteria_thresh = 0.0001
# channel = 1
# time_points = [0]
# dask_label = True

# 5x
sigma_blur = 1.0
bright_thresh = 0.01
sigma_low = 0.5
sigma_high = 2.3
disk_size = 1
bacteria_thresh = 0.0001
channel = 0
time_points = [0]
dask_label = True
sigma_blur_agg = 3
disk_size_agg = 3
bacteria_thresh_agg = 10 ** -2.3    #-2.45

file_name = 'im.ome.zarr/0'
for path in experiment_paths:
    larvae_dirs = glob(path + '/larva*')
    for larvae_dir in larvae_dirs:
        if larvae_dir == '/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_06_04-NP1029-Gal4_UAS-Mhc-RNAi_ecoli-hs-dtom_4hrs/no_heartbeat/larva_2':
            continue
        print(larvae_dir)
        path_to_zarr = larvae_dir + '/' + file_name
        if len(glob(path_to_zarr)) == 0:
            continue
        path_to_output_labels = larvae_dir + '/bacteria.segmentation.zarr'
        if len(glob(path_to_output_labels)) > 0:
            print('segmentation file already exists! skipping')
            continue
        segment_bacteria(path_to_zarr, path_to_output_labels, bright_thresh=bright_thresh, sigma_blur=sigma_blur, sigma_low=sigma_low,
                         sigma_high=sigma_high, disk_size=disk_size, bacteria_thresh=bacteria_thresh,
                         time_points=time_points, maximum_size=None,
                         channel=channel, dask_label=dask_label, sigma_blur_agg=sigma_blur_agg, disk_size_agg=disk_size_agg, bacteria_thresh_agg=bacteria_thresh_agg)
