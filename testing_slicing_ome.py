import zarr
from readwrite import get_nonzero_slicing_range_ome

path_to_zarr = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_4/tmp.zarr'
group_name = 'tmp_big_stack'

slicing = get_nonzero_slicing_range_ome(path_to_zarr, group_name)