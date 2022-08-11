from readwrite import crop_padding
import zarr

path_to_tmp_zarr = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_4/tmp.zarr'
slicing = ((239, 514), (0, 7723), (1327, 3875))
crop_padding(path_to_tmp_zarr, slicing=slicing)