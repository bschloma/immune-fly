from readwrite import create_pyramid_from_zarr

path_plain_zarr = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_4/tmp.crop.zarr'
group_name = 'tmp_crop'
path_to_new_zarr = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_4/crop.ome.zarr'
pyramid_scales = 5


root = create_pyramid_from_zarr(path_plain_zarr, group_name, path_to_new_zarr, pyramid_scales)