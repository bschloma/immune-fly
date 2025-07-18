from segmentation import segment_nuclei
import zarr
from zarr.storage import DirectoryStore
import napari
import numpy as np
import dask.array as da
from glob import glob


experiment_paths = [
    r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_06_ca-Gal4_UAS-His-RFP_mNG-EcR-B1']

sigma = 1.0
thresh = 10 ** -2.21
sigma_low = 8
opening_size = 19
time_points = [0]
dask_label = True

file_name = 'im.ome.no_correction.zarr'
for path in experiment_paths:
    larvae_dirs = glob(path + '/larva*')
    for larvae_dir in larvae_dirs:
        path_to_ome_zarr = larvae_dir + '/' + file_name
        path_to_output_labels = larvae_dir + '/segmentation.zarr'
        if len(glob(path_to_output_labels)) > 0:
            continue
        print(larvae_dir)
        segment_nuclei(path_to_ome_zarr, path_to_output_labels, sigma, thresh, time_points, sigma_low=sigma_low, opening_size=opening_size)


