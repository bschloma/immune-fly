import numpy as np
import dask.array as da
from dask.diagnostics import ProgressBar
from dexp.datasets import ZDataset
import napari
from segmentation import run_2D_watershed

path_to_ds = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_1/crop_chk.zarr'
ds = ZDataset(path_to_ds, mode='r')
mem = ds.get_array('mem-green')
#mem_da = da.from_array(mem[0], chunks=(1, 8200, 2071))
mem_da = da.from_array(mem[0, 200:220], chunks=(1, 2048, 2071))
out_zarr = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_1/frangi_seg.zarr'

labels = run_2D_watershed(mem_da)

with ProgressBar():
    #labels = labels.compute()
    labels.to_zarr(out_zarr)

# view result
viewer = napari.view_image(labels)
viewer.add_image(mem_da)
#dpt = ds.get_array('dpt-red')
#dpt_da = da.from_array(dpt[0], chunks=(1, 1024, 2071))
#viewer.add_image(dpt_da)
#viewer = napari.view_image(mem_da)

