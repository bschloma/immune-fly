import numpy as np
import cupy as cp
from cucim.skimage.measure import regionprops
import dask.array as da
from dask_image.ndmeasure import area
from dask.diagnostics import ProgressBar
import zarr
import napari
from segmentation import filter_segments_by_size_2D

path_to_labels = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_1/labels2D.zarr'
z = zarr.open(path_to_labels, 'r')
labels = da.from_array(z, chunks=(1, 1024, 2071))
min_area = 100
max_area = 1000

#filtered_labels = filter_segments_by_size_2D(labels, min_area, max_area)

#with ProgressBar():
#    filtered_labels = filtered_labels.compute()

labels_cu = labels.map_blocks(cp.asarray, meta=labels, dtype=np.int32)
props = da.map_blocks(regionprops, labels_cu, dtype=np.int32)
props_np = props.map_blocks(cp.asnumpy, meta=props, dtype=np.float32)

with ProgressBar():
    props_np = props_np.compute()

#props = area(labels, labels, index=np.uint16(da.unique(labels)))

#with ProgressBar():
#    props_np = props.compute()
# view result
#viewer = napari.view_image(filtered_labels)
#viewer.add_image(mem_da)
#dpt = ds.get_array('dpt-red')
#dpt_da = da.from_array(dpt[0], chunks=(1, 1024, 2071))
#viewer.add_image(dpt_da)

