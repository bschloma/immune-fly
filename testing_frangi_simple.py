import numpy as np
import dask.array as da
from skimage.filters import gaussian, frangi
from dexp.datasets import ZDataset
import napari


path_to_ds = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_1/crop_chk.zarr'
ds = ZDataset(path_to_ds, mode='r')
sigma = 1.0

#mem = da.from_array(ds.get_array('mem-green')[0, 0:10], chunks=(64, 256, 256))
mem = da.from_array(ds.get_array('mem-green')[0], chunks=(1, 8200, 2071))


def func(arr, sigma):
    arr = arr[0]
    arr = frangi(arr, sigma)
    arr = np.expand_dims(arr, axis=0)
    #print(arr.shape)
    return arr


filt = mem.map_blocks(func, sigma, dtype=np.float32)
#filt = da.map_overlap(func, mem, np.array(sigma), depth=1, dtype=np.float32)

# filt = da.zeros_like(mem)
# for z in range(mem.shape[0]):
#     print(z)
#     this_slice = mem[z]
#     filt[z] = frangi(this_slice, sigma)


viewer = napari.view_image(filt)
#filt_np = filt.compute()
