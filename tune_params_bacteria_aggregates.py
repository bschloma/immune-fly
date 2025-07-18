import cupy as cp
import numpy as np
import dask.array as da
import zarr
from cucim.skimage.filters import gaussian, frangi, median, sobel, difference_of_gaussians
from cucim.skimage.morphology import binary_opening, binary_erosion, disk, white_tophat, binary_closing, remove_small_holes
from cucim.skimage.measure import label
from cucim.skimage.segmentation import morphological_geodesic_active_contour
from dexp.datasets import ZDataset
import time
import napari
from magicgui import magicgui
from napari.types import ImageData
from PIL import Image
from scipy.ndimage import distance_transform_edt
from zarr.storage import DirectoryStore


# sigma_blur = 8.0
# sigmas = 8.0#np.arange(1, 10, 2)
# alpha = 0.5
# beta = 0.5
# gamma = 1E-4


def filter(arr, sigma_blur, disk_size=5, l_bacteria_thresh=-1.8):
    # unpack this zslice
    arr = arr[0]

    # apply some filters
    arr = gaussian(arr, sigma=sigma_blur)

    arr = arr > 10 ** l_bacteria_thresh

    arr = binary_opening(arr, disk(disk_size))

    # reshape into 3D arr
    arr = cp.expand_dims(arr, axis=0)

    return arr


def to_gpu(arr):
    return cp.asarray(arr)


def gpu_process(darr, sigma_blur=1.0, disk_size=5, l_bacteria_thresh=-1.8):
    # lazy move to gpu
    mem_cu = darr.map_blocks(to_gpu, dtype=np.float32)

    # lazy apply filter
    filt = da.map_blocks(filter, mem_cu, sigma_blur, disk_size, l_bacteria_thresh, dtype=np.float32)

    # actual compute step for visualization
    filt_np = filt.map_blocks(cp.asnumpy, meta=filt, dtype=np.float32)
    #filt_np = filt.map_blocks(cp.asnumpy, meta=filt, dtype=np.uint16)


    return filt_np

@magicgui(
    auto_call=True,
    sigma_blur={"widget_type": "FloatSlider", "max": 49, "min": 0.5},
    disk_size={"widget_type": "Slider", "max": 9},
    l_bacteria_thresh={"widget_type": "FloatSlider", "max": -1, "min": -4},
    layout='vertical'
)
def magic_func(layer: ImageData, sigma_blur: float = 1.0, disk_size: int = 1, l_bacteria_thresh: float = -3) -> ImageData:
    if layer is not None:
        return gpu_process(layer, sigma_blur, disk_size, l_bacteria_thresh)



#path_to_zarr = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_06_ca-Gal4_UAS-His-RFP_mNG-EcR-B1/larva_2/im.ome.no_correction.zarr/0'
path_to_zarr = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_05_30-dpt-gfp_r4-gal4_ecoli-hs-dtom_4hrs_flow_field/larva_1/im.ome.zarr/0'
im_da = da.from_zarr(DirectoryStore(path_to_zarr))[0, 0]

# then view with napari
viewer = napari.view_image(im_da)

# Add it to the napari viewer
viewer.window.add_dock_widget(magic_func)
# update the layer dropdown menu when the layer list changes
viewer.layers.events.changed.connect(magic_func.reset_choices)

napari.run()

