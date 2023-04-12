import cupy as cp
import numpy as np
import dask.array as da
import zarr
from cucim.skimage.filters import gaussian, frangi, median, sobel
from cucim.skimage.morphology import binary_opening, binary_erosion, disk, white_tophat
from cucim.skimage.measure import label
from cucim.skimage.segmentation import morphological_geodesic_active_contour
from dexp.datasets import ZDataset
import time
import napari
from magicgui import magicgui
from napari.types import ImageData
from PIL import Image
from scipy.ndimage import distance_transform_edt


path_to_mip = r'/media/brandon/Data1/Brandon/fly_immune/Serenity/2022_08_24_dpt-gfp_silverman_r4-gal4_uas-mcd8-mcherry_ecoli-hs-dtom/larvae_1/ds_h5/scan_0/pred_mip.tif'
#path_to_ds = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_4/ds_h5/scan_0/predictions.zarr'
# sigma_blur = 8.0
# sigmas = 8.0#np.arange(1, 10, 2)
# alpha = 0.5
# beta = 0.5
# gamma = 1E-4


def frangi_filter(arr, _sigma_blur, _sigmas, _alpha, _beta, _gamma, thresh):
    # unpack this zslice
    arr = arr[0]
    #og_arr = arr

    # apply some filters
    arr = gaussian(arr, sigma=_sigma_blur)
    if _sigmas > 0:
        arr = frangi(arr, sigmas=_sigmas, alpha=_alpha, beta=_beta, gamma=_gamma)

    # create and apply a mask
    arr = arr < thresh
    #arr = arr > thresh
    #arr = arr * (og_arr > 3000)
    #arr = binary_opening(arr, disk(9))
    #arr = binary_erosion(arr, disk(5))
    #arr = arr == 0

    #arr = arr * og_arr
    arr = label(arr)

    # reshape into 3D arr
    arr = cp.expand_dims(arr, axis=0)

    return arr


def to_gpu(arr):
    return cp.asarray(arr)


def gpu_process(darr, sigma_blur=1.0, beta=0.5, l_gamma=-0.3, l_thresh=-12.0, sigmas=8.0):
    #sigma_blur = 3.0
    #sigmas = 8.0  # np.arange(1, 10, 2)
    alpha = 0.5
    # lazy move to gpu
    mem_cu = darr.map_blocks(to_gpu, dtype=np.float32)

    # lazy apply filter
    filt = da.map_blocks(frangi_filter, mem_cu, sigma_blur, sigmas, alpha, beta, _gamma=10**l_gamma, thresh=10**l_thresh, dtype=np.float32)

    # actual compute step for visualization
    #filt_np = filt.map_blocks(cp.asnumpy, meta=filt, dtype=np.float32)
    filt_np = filt.map_blocks(cp.asnumpy, meta=filt, dtype=np.uint16)


    return filt_np

@magicgui(
    auto_call=True,
    sigma_blur={"widget_type": "FloatSlider", "max": 9, "min": 0.5},
    beta={"widget_type": "FloatSlider", "max": 1},
    l_gamma={"widget_type": "FloatSlider", "max": 0, "min": -6},
    l_thresh={"widget_type": "FloatSlider", "max": 0.5, "min": -20},
    sigmas={"widget_type": "FloatSlider", "max": 50.0, "min": 0},
    layout='vertical'
)
def magic_func(layer: ImageData, sigma_blur: float = 1.0, beta: float = 0.5, l_gamma: float = -0.3, l_thresh: float = -12.0, sigmas: float = 8.0) -> ImageData:
    if layer is not None:
        return gpu_process(layer, sigma_blur, beta, l_gamma, l_thresh, sigmas)


#mem = zarr.open(path_to_mip, 'r')
#ds = ZDataset(path_to_ds, 'r')
#mem = ds.get_array('mem-green')
#mem = zarr.open(path_to_ds).pred
#mem_da = da.from_array(mem[0, 0], chunks=mem.chunks[2:])
#mem_da = da.from_array(mem, chunks=(1, 8200, 2071))
#mem_da = da.from_array(mem[0], chunks=(1, 1024, 1024))
mem = np.array(Image.open(path_to_mip))
#mem = mem > 0.4
#mem = distance_transform_edt(mem)
mem = np.expand_dims(mem, axis=0)
mem_da = da.from_array(mem)

#filt_np = gpu_process(mem_da, beta)



#out_zarr = zarr.open(r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_1/frangi.zarr', 'w')
#filt_np.to_zarr(r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_1/frangi.zarr')
#final = filt_np * mem_da
# then view with napari
viewer = napari.view_image(mem_da)

# Add it to the napari viewer
viewer.window.add_dock_widget(magic_func)
# update the layer dropdown menu when the layer list changes
viewer.layers.events.changed.connect(magic_func.reset_choices)

napari.run()

