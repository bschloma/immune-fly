import cupy as cp
import numpy as np
import dask.array as da
import zarr
from cucim.skimage.filters import gaussian, frangi, median, sobel
from cucim.skimage.segmentation import morphological_geodesic_active_contour
from dexp.datasets import ZDataset
import time
import napari

# mempool = cp.get_default_memory_pool()
# with cp.cuda.Device(0):
#     mempool.set_limit(size=5*1024**3)  # 5 GiB

path_to_ds = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_1/crop_chk.zarr'
ds = ZDataset(path_to_ds, mode='r')
sigma_blur = 8.0
sigmas = 8.0#np.arange(1, 10, 2)
alpha = 0.5
beta = 0.5
gamma = 1E-4


def test_fcn(arr, _sigma_blur, _sigmas, _alpha, _beta, _gamma):
    # do something
    arr = arr[0]
    # maybe median filter?
    arr = gaussian(arr, sigma=_sigma_blur)
    #arr = median(arr)
    #filt_arr = frangi(arr, sigmas=_sigmas, alpha=_alpha, beta=_beta, gamma=_gamma)
    #arr = sobel(arr)
    #filt_arr = filt_arr / cp.max(arr)
    #arr = morphological_geodesic_active_contour(arr, num_iter=100, init_level_set=frangi(arr, sigmas=_sigmas, alpha=_alpha, beta=_beta, gamma=_gamma))
    #filt_arr = 1 - filt_arr
    #filt_arr = arr *filt_arr
    arr = cp.expand_dims(arr, axis=0)

    return arr


def to_gpu(arr):
    return cp.asarray(arr)


mem = ds.get_array('mem-green')
mem_da = da.from_array(mem[0], chunks=(1, 8200, 2071))
#mem_da = da.from_array(mem[0], chunks=(1, 1024, 1024))

mem_cu = mem_da.map_blocks(to_gpu, dtype=np.float32)

filt = da.map_blocks(test_fcn, mem_cu, sigma_blur, sigmas, alpha, beta, gamma, dtype=np.float32)

filt_np = filt.map_blocks(cp.asnumpy, meta=filt, dtype=np.float32)

#out_zarr = zarr.open(r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_1/frangi.zarr', 'w')
#filt_np.to_zarr(r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_1/frangi.zarr')
#final = filt_np * mem_da
# then view with napari
viewer = napari.view_image(filt_np)


