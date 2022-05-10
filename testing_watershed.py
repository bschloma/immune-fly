import cupy as cp
import numpy as np
import dask.array as da
from dask.diagnostics import ProgressBar
from cucim.skimage.filters import gaussian, frangi
from cucim.skimage.measure import label
from cucim.skimage.morphology import binary_opening, binary_erosion, disk, white_tophat
from dexp.datasets import ZDataset
from skimage.segmentation import watershed
from dask_image.ndfilters import gaussian_filter
import time
import napari

path_to_ds = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_1/crop_chk.zarr'
ds = ZDataset(path_to_ds, mode='r')


def frangi_filter(arr, _sigma_blur, _sigmas, _alpha, _beta, _gamma, thresh):
    # # unpack this zslice
    # arr = arr[0]
    # og_arr = arr
    #
    # # apply some filters
    # arr = gaussian(arr, sigma=_sigma_blur)
    # arr = frangi(arr, sigmas=_sigmas, alpha=_alpha, beta=_beta, gamma=_gamma)
    #
    # # create and apply a mask
    # arr = arr < thresh
    # arr = arr * (og_arr > 2000)
    # arr = arr * og_arr
    # arr = gaussian(arr, sigma=_sigma_blur)
    #
    # # reshape into 3D arr
    # arr = cp.expand_dims(arr, axis=0)

    # unpack this zslice
    arr = arr[0]
    og_arr = arr

    # apply some filters
    arr = gaussian(arr, sigma=_sigma_blur)
    arr = frangi(arr, sigmas=_sigmas, alpha=_alpha, beta=_beta, gamma=_gamma)

    # create and apply a mask
    # arr = arr < thresh
    arr = arr > thresh
    arr = arr * (og_arr > 3000)
    arr = binary_opening(arr, disk(9))
    # arr = binary_erosion(arr, disk(5))

    # arr = arr * og_arr
    arr = label(arr)

    # reshape into 3D arr
    arr = cp.expand_dims(arr, axis=0)

    return arr


def to_gpu(arr):
    return cp.asarray(arr)


def gpu_process(darr, beta=0.5, l_gamma=-0.3, l_thresh=-12.0):
    sigma_blur = 3.0
    sigmas = 20.0  # np.arange(1, 10, 2)
    alpha = 0.5
    # lazy move to gpu
    mem_cu = darr.map_blocks(to_gpu, dtype=np.float32)

    # lazy apply filter
    filt = da.map_blocks(frangi_filter, mem_cu, sigma_blur, sigmas, alpha, beta, _gamma=10**l_gamma, thresh=10**l_thresh, dtype=np.float32)

    # actual compute step for visualization
    filt_np = filt.map_blocks(cp.asnumpy, meta=filt, dtype=np.float32)

    return filt_np


mem = ds.get_array('mem-green')
#mem_da = da.from_array(mem[0], chunks=(1, 8200, 2071))
mem_da = da.from_array(mem[0], chunks=(1, 1024, 2071))

result = gpu_process(mem_da, beta=1.0, l_gamma=-6.0, l_thresh=-12.0)
#result = result.rechunk(chunks=(64, 512, 512))
#labels = result.map_blocks(watershed)
labels = da.map_blocks(watershed, mem_da, result)

#mem_da = gaussian_filter(da.from_array(mem[0], chunks=(1, 1024, 2071)), sigma=3.0)

#labels = mem_da.map_blocks(watershed)
with ProgressBar():
    labels = labels.compute()

# view result
viewer = napari.view_image(labels)
viewer.add_image(mem_da)
dpt = ds.get_array('dpt-red')
dpt_da = da.from_array(dpt[0], chunks=(1, 1024, 2071))
viewer.add_image(dpt_da)