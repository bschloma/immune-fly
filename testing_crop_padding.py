from readwrite import crop_padding
import zarr

path_to_tmp_zarr = r'/media/brandon/Data1/Brandon/fly_immune/Serenity/2022_08_24_dpt-gfp_silverman_r4-gal4_uas-mcd8-mcherry_ecoli-hs-dtom/larvae_1/tmp.zarr'
slicing = ((236, 475), (0, 8944), (1920, 4650))
crop_padding(path_to_tmp_zarr, slicing=slicing)
