import numpy as np
import dask.array as da
from dask.diagnostics import ProgressBar
from dexp.datasets import ZDataset
import napari
import zarr
from skimage.segmentation import watershed

#path_to_ds = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_1/crop_chk.zarr'
#ds = ZDataset(path_to_ds, mode='r')
#mem = ds.get_array('mem-green')
#mem_da = da.from_array(mem[0], chunks=(1, 8200, 2071))
#mem_da = da.from_array(mem[0], chunks=(1, 8200, 2071))
labels_zarr = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ctl_no_ether_no_inj/larvae_1/ds_h5/scan_0/segmentation_mip_manual.zarr'
boundaries_path = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ctl_no_ether_no_inj/larvae_1/ds_h5/scan_0/predictions_mip.zarr'
seeds_path =r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ctl_no_ether_no_inj/larvae_1/ds_h5/scan_0/manual_seeds.zarr'
#seeds_path = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_1/ds_h5/scan_0/manual_seeds_mip.zarr'


boundaries = da.from_zarr(boundaries_path, chunks=(1, 7723, 2548))
seeds = da.from_zarr(seeds_path, chunks=(1, 7723, 2548))

#labels = da.map_overlap(watershed, boundaries, seeds, depth={0: 0, 1: 1, 2: 0})
labels = da.map_blocks(watershed, boundaries, seeds)

with ProgressBar():
    labels.to_zarr(labels_zarr)


# # view result
# labels = zarr.open(labels_zarr, 'r')
# viewer = napari.view_image(labels)
# viewer.add_image(mem_da)
# dpt = ds.get_array('dpt-red')
# dpt_da = da.from_array(dpt[0], chunks=(1, 1024, 2071))
# viewer.add_image(dpt_da)
#viewer = napari.view_image(mem_da)

