""""""


from readwrite import convert_czi_views_to_zarr


path_to_czi_dir = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_1/full_scan'
path_to_new_zarr = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_1/full_scan.zarr'
core_file_name = r'full_scan'
suffix = r'.czi'
namestr = '*.czi'
chunk_sizes = (1, 64, 256, 256)
channel_names = ['488nm', '561nm']

convert_czi_views_to_zarr(path_to_czi_dir, path_to_new_zarr, namestr, core_file_name, suffix, chunk_sizes, channel_names)


