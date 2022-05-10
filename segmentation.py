import numpy as np
import pandas as pd
import cupy as cp
import dask.array as da
from dask.diagnostics import ProgressBar
from dask_image.ndfilters import gaussian_filter
from dask_image.ndmeasure import area
from cucim.skimage.filters import gaussian, frangi
from cucim.skimage.measure import label
from cucim.skimage.morphology import binary_opening, binary_erosion, disk, white_tophat
from dexp.datasets import ZDataset
from skimage.segmentation import watershed
from skimage.measure import label as sk_label
#from scipy.ndimage import distance_transform_edt, label
import zarr


def init_frangi_params():
    params = pd.DataFrame()
    params.sigma_blur = 3.0
    params.sigmas = 8.0
    params.alpha = 0.5
    params.beta = 0.8
    params.gamma = 0.8

    return params


def cp_make_boundaries(arr, method="frangi", params=init_frangi_params()):
    """ arr is a ZYX cupy array """
    # unpack this zslice
    arr = arr[0]

    if method == "frangi":
        # apply some filters
        arr = gaussian(arr, sigma=params.sigma_blur)
        arr = frangi(arr, sigmas=params.sigmas, alpha=params.alpha, beta=params.beta, gamma=params.gamma)

        # scale 0 to 1
        arr = (arr - cp.min(arr)) / (cp.max(arr) - cp.min(arr))
    else:
        raise NotImplementedError

    # reshape into 3D arr
    arr = cp.expand_dims(arr, axis=0)

    return arr


def cp_make_seeds(arr, method="frangi", params=init_frangi_params(), thresh=1E-6):
    """gpu based method for seeds. inverse frangi mask"""
    # unpack this zslice
    arr = arr[0]

    if method == "frangi":
        og_arr = arr
        # apply some filters
        arr = gaussian(arr, sigma=params.sigma_blur)
        arr = frangi(arr, sigmas=params.sigmas, alpha=params.alpha, beta=params.beta, gamma=params.gamma)

        # create and apply a mask
        arr = arr > thresh
        arr = arr * (og_arr > 3000)
        arr = binary_opening(arr, disk(9))

        # arr = arr * og_arr
        arr = label(arr)

    else:
        raise NotImplementedError

    # reshape into 3D arr
    arr = cp.expand_dims(arr, axis=0)

    return arr


def make_seeds_distance(arr, seed_inten_thresh=0.4, seed_distance_thresh=0.5):
    """use dask around scipy.ndimage function. np backed arrays"""
    #arr = gaussian_filter(arr, sigma=2.0)
    arr = (arr - da.min(arr)) / (da.max(arr) - da.min(arr))
    arr = arr < seed_inten_thresh
    arr = distance_transform_edt(arr)
    #arr = (arr - da.min(arr)) / (da.max(arr) - da.min(arr))
    #arr = arr > seed_distance_thresh
    #arr = label(arr)

    return arr


def to_gpu(arr):
    return cp.asarray(arr)


def run_2D_watershed(arr, method="frangi", params=init_frangi_params(), thresh=1E-6):
    """main watershed fcn. arr is a ZYX dask array"""
    # boundaries: use gpu
    # arr_cu = arr.map_blocks(to_gpu, dtype=np.float32)
    # boundaries = da.map_blocks(cp_make_boundaries, arr_cu, method, params, dtype=np.float32)
    # boundaries = boundaries.map_blocks(cp.asnumpy, meta=boundaries, dtype=np.float32)
    # seeds = da.map_blocks(cp_make_seeds, arr_cu, method, params, thresh, dtype=np.float32)
    # seeds = seeds.map_blocks(cp.asnumpy, meta=seeds, dtype=np.float32)
    # labels = da.map_blocks(watershed, boundaries, seeds)

    arr_cu = arr.map_overlap(to_gpu, dtype=np.float32, depth={0: 0, 1: 1, 2: 0})
    boundaries = da.map_overlap(cp_make_boundaries, arr_cu, dtype=np.float32, depth={0: 0, 1: 1, 2: 0}, method=method, params=params)
    boundaries = boundaries.map_overlap(cp.asnumpy, meta=boundaries, dtype=np.float32, depth={0: 0, 1: 1, 2: 0})
    seeds = da.map_overlap(cp_make_seeds, arr_cu, dtype=np.float32, depth={0: 0, 1: 1, 2: 1}, method=method, params=params, thresh=thresh)
    seeds = seeds.map_overlap(cp.asnumpy, meta=seeds, dtype=np.float32, depth={0: 0, 1: 1, 2: 0})
    labels = da.map_overlap(watershed, boundaries, seeds, depth={0: 0, 1: 1, 2: 0})

    return labels


def filter_segments_by_size_2D(labels, min_area, max_area):
    """dask array input"""
    ids = da.unique(labels)
    areas = area(labels, labels, index=np.uint16(ids))
    bad_ids = ids[[areas < min_area] or [areas > max_area]]
    for bad_id in bad_ids:
        labels[labels == bad_id] = 0

    return labels


def make_boundaries(arr, method="frangi", params=init_frangi_params()):
    """run just the boundary prediction. inputs and outputs are dask arrays"""
    arr_cu = arr.map_blocks(to_gpu, dtype=np.float32)
    boundaries = da.map_blocks(cp_make_boundaries, arr_cu, method, params, dtype=np.float32)
    boundaries = boundaries.map_blocks(cp.asnumpy, meta=boundaries, dtype=np.float32)

    return boundaries


def make_seeds(arr, method="frangi", params=init_frangi_params(), thresh=1E-6):
    """run just seeds. inputs and outputs are dask arrays"""
    arr_cu = arr.map_blocks(to_gpu, dtype=np.float32)
    seeds = da.map_blocks(cp_make_seeds, arr_cu, method, params, thresh, dtype=np.float32)
    seeds = seeds.map_blocks(cp.asnumpy, meta=seeds, dtype=np.float32)

    return seeds


def run_ws_from_boundaries(arr, boundaries, method="frangi", params=init_frangi_params(), thresh=1E-6):
    """pass boundaries as input. inputs and outputs are dask arrays"""
    arr_cu = arr.map_blocks(to_gpu, dtype=np.float32)
    seeds = da.map_blocks(cp_make_seeds, arr_cu, method, params, thresh, dtype=np.float32)
    seeds = seeds.map_blocks(cp.asnumpy, meta=seeds, dtype=np.float32)
    labels = da.map_blocks(watershed, boundaries, seeds)

    return labels


def simple_make_seeds_from_boundaries(boundaries):
    seeds = sk_label(boundaries < 0.1)

    return seeds


def make_seeds_arr_from_points(points_df, path_to_seeds_zarr, shape):
    seeds = zarr.zeros(shape, store=path_to_seeds_zarr)
    for i in range(len(points_df)):
        seeds[np.uint16(points_df[0][i]), np.uint16(points_df[1][i]), np.uint16(points_df[2][i])] = i + 1

    return


class WS:
    """creating this weird class to use as superpixel generator in elf gasp"""
    def __init__(self, seeds, method="frangi", params=init_frangi_params(), thresh=1E-6):

        self.method = method
        self.params = params
        self.thresh = thresh
        self.seeds = seeds

    def __call__(self, affinities, foreground_mask=None):
        boundaries = 1 - affinities[0]
        #labels = da.map_blocks(watershed, boundaries, self.seeds)
        seeds = simple_make_seeds_from_boundaries(boundaries)
        labels = watershed(boundaries, seeds)

        return labels





