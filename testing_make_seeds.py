import numpy as np
import cupy as cp
import dask.array as da
from dask.diagnostics import ProgressBar
from dexp.datasets import ZDataset
import napari
from segmentation import cp_make_seeds, to_gpu
from dask_image.ndfilters import gaussian_filter


path_to_ds = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_1/crop_chk.zarr'
ds = ZDataset(path_to_ds, mode='r')
mem = ds.get_array('mem-green')
#mem_da = da.from_array(mem[0], chunks=(1, 8200, 2071))
arr = da.from_array(mem[0], chunks=(1, 1024, 2071))
arr_cu = arr.map_blocks(to_gpu, dtype=np.float32)
seeds = da.map_blocks(cp_make_seeds, arr_cu, dtype=np.float32)
seeds = seeds.map_blocks(cp.asnumpy, meta=seeds, dtype=np.float32)

with ProgressBar():
    seeds = seeds.compute()

# view result
viewer = napari.view_image(seeds)
#viewer.add_image(mem_da)
#dpt = ds.get_array('dpt-red')
#dpt_da = da.from_array(dpt[0], chunks=(1, 1024, 2071))
#viewer.add_image(dpt_da)

