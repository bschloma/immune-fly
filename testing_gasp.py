import zarr
import numpy as np
from elf.segmentation import run_GASP, GaspFromAffinities
from dexp.datasets import ZDataset
import dask.array as da
from dask.diagnostics import ProgressBar
from segmentation import make_boundaries
import napari
import matplotlib.pyplot as plt


def run_gasp(affs):
    seg, runtime = gasp_instance(affs)

    return seg


path_to_ds = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_1/crop_chk.zarr'
ds = ZDataset(path_to_ds, mode='r')
mem = ds.get_array('mem-green')
#mem_da = da.from_array(mem[0], chunks=(1, 8200, 2071))
mem_da = da.from_array(mem[0, 270:280, 500:1000, 500:1000], chunks=(1, 512, 512))

affinities = 1 - make_boundaries(mem_da)
affinities = da.stack([affinities, affinities, affinities], axis=0)
offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
run_GASP_kwargs = {'linkage_criteria': 'average', 'add_cannot_link_constraints': False}
gasp_instance = GaspFromAffinities(offsets, run_GASP_kwargs=run_GASP_kwargs)

res = da.map_blocks(run_gasp, affinities, meta=affinities, dtype=affinities.dtype, drop_axis=0, chunks=mem_da.chunks)

with ProgressBar():
    res_out = res.compute()

viewer = napari.view_labels(res_out)
viewer.add_image(mem_da)
#viewer = napari.view_image(mem_da)
