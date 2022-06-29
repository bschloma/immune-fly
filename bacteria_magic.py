import cupy as cp
import numpy as np
import dask.array as da
import zarr
from cucim.skimage.filters import gaussian, frangi, median, sobel, difference_of_gaussians
from cucim.skimage.morphology import binary_opening, binary_erosion, disk, white_tophat
from cucim.skimage.measure import label
from cucim.skimage.segmentation import morphological_geodesic_active_contour
from dexp.datasets import ZDataset
import time
import napari
from magicgui import magicgui
from napari.types import ImageData


def highlight_bacteria(arr, _sigma_blur, _fb_thresh, _sigma_low, _sigma_high, _disk_size, _bacteria_thresh):
    # unpack this zslice
    arr = arr[0]

    # apply some filters
    arr = gaussian(arr, sigma=_sigma_blur)
    arr[arr > _fb_thresh] = 0
    arr = difference_of_gaussians(arr, _sigma_low, _sigma_high)

    if _disk_size > 0:
        arr = white_tophat(arr, disk(_disk_size))

    arr = arr > _bacteria_thresh
    arr = label(arr)

    # reshape into 3D arr
    arr = cp.expand_dims(arr, axis=0)

    return arr


def to_gpu(arr):
    return cp.asarray(arr)


def gpu_process(darr, sigma_blur=1.0, thresh=0.04, sigma_low=0.5, sigma_high=1.0, disk_size=3, bacteria_thresh=0.0005):
    # lazy move to gpu
    mem_cu = darr.map_blocks(to_gpu, dtype=np.float32)

    # lazy apply filter
    filt = da.map_blocks(highlight_bacteria, mem_cu, sigma_blur, thresh, sigma_low, sigma_high, disk_size, bacteria_thresh, dtype=np.float32)

    # actual compute step for visualization
    #filt_np = filt.map_blocks(cp.asnumpy, meta=filt, dtype=np.float32)
    filt_np = filt.map_blocks(cp.asnumpy, meta=filt, dtype=np.uint16)


    return filt_np

@magicgui(
    auto_call=True,
    sigma_blur={"widget_type": "FloatSlider", "max": 9, "min": 0.5},
    thresh={"widget_type": "FloatSlider", "max": 1.0, "min": 0.0},
    sigma_low={"widget_type": "FloatSlider", "max": 1.9, "min": 0.5},
    sigma_high={"widget_type": "FloatSlider", "max": 9.0, "min": 2.0},
    disk_size={"widget_type": "Slider", "max": 9, "min": 0},
    bacteria_thresh={"widget_type": "FloatSlider", "max": 0.001, "min": 0.0001},
    layout='vertical'
)
def magic_func(layer: ImageData, sigma_blur: float = 1.0, thresh: float = 0.04, sigma_low: float = 1.0, sigma_high: float = 2.0, disk_size: int = 3, bacteria_thresh: float = 0.0005) -> ImageData:
    if layer is not None:
        return gpu_process(layer, sigma_blur, thresh, sigma_low, sigma_high, disk_size, bacteria_thresh)


path_to_ds = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_4/crop.zarr'
ds = ZDataset(path_to_ds, 'r')
mem = ds.get_array('mem-green')
mem_da = da.from_array(mem[0], chunks=mem.chunks[1:])

# launch napari
viewer = napari.view_image(mem_da)

# Add it to the napari viewer
viewer.window.add_dock_widget(magic_func)
# update the layer dropdown menu when the layer list changes
viewer.layers.events.changed.connect(magic_func.reset_choices)

napari.run()

