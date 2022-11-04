from readwrite import create_pyramid_from_zarr

path_plain_zarr =  r'/media/brandon/Data1/Brandon/fly_immune/Serenity/2022_09_23_dpt-gfp_ecoli-hs-dtom_timeseries/larvae_1/tmp.crop.zarr'
group_name = 'tmp_crop'
path_to_new_zarr =  r'/media/brandon/Data1/Brandon/fly_immune/Serenity/2022_09_23_dpt-gfp_ecoli-hs-dtom_timeseries/larvae_1/crop.ome.zarr'
pyramid_scales = 5


root = create_pyramid_from_zarr(path_plain_zarr, group_name, path_to_new_zarr, pyramid_scales)