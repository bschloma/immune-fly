from segmentation import segment_nuclei
import zarr
from zarr.storage import DirectoryStore
import napari
import numpy as np
import dask.array as da


path_to_ome_zarr = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_06_ca-Gal4_UAS-His-RFP_mNG-EcR-B1/larva_4/im.ome.no_correction.zarr'
path_to_output_labels =r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_06_ca-Gal4_UAS-His-RFP_mNG-EcR-B1/larva_4/segmentation.zarr'

# timeseries params
# sigma = 1.93
# thresh = 10 ** -2.0
# time_points = np.arange(90)

sigma = 1.0
thresh = 10 ** -2.21
sigma_low = 8
opening_size = 19
time_points = [0]
dask_label = True

segment_nuclei(path_to_ome_zarr, path_to_output_labels, sigma, thresh, time_points, sigma_low=sigma_low, opening_size=opening_size, dask_label=dask_label)
seg = zarr.open(DirectoryStore(path_to_output_labels), 'r')
im = da.from_zarr(zarr.open(DirectoryStore(path_to_ome_zarr + '/0')))[np.array(time_points), 1]
viewer = napari.view_image(im)
viewer.add_labels(seg)
