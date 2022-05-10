import cupy as cp
import numpy as np
import dask.array as da
import napari
from cucim.skimage.filters import gaussian

sigma = 0.1

def test_fcn(arr, _sigma_blur):
    # do something
    arr = arr[0]
    for i in range(2):
        arr = gaussian(arr, sigma=_sigma_blur)
    #arr = frangi(arr, sigmas=_sigmas, alpha=_alpha, beta=_beta, gamma=_gamma)
    arr = cp.expand_dims(arr, axis=0)

    return arr


def to_gpu(arr):
    return cp.asarray(arr)


grid = da.random.random((10, 1000, 1000), chunks=(1, 1000, 1000))

grid_cu = grid.map_blocks(to_gpu, dtype=np.float32)

sim = da.map_blocks(test_fcn, grid, sigma, dtype=np.float32)

sim_np = sim.map_blocks(cp.asnumpy, meta=sim, dtype=np.float32)

# then view with napari
viewer = napari.view_image(sim_np)


