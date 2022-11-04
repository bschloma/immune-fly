from readwrite import crop_padding
import zarr

path_to_tmp_zarr = r'/media/brandon/Data1/Brandon/fly_immune/Serenity/2022_09_23_dpt-gfp_ecoli-hs-dtom_timeseries/larvae_1/tmp.zarr'
slicing = ((269, 636), (0, 7509), (1920, 4550))
crop_padding(path_to_tmp_zarr, slicing=slicing)
