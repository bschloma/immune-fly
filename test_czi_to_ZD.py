""""""


from readwrite import convert_czi_views_to_fuse_reg_ZD

path_to_czi_dir = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_4/full_scan'
path_to_new_zarr = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_4/test_big.zarr'
core_file_name = r'full_scan'
suffix = r'.czi'
namestr = '*.czi'
chunk_sizes = (1, 202, 1920, 1920)#(1, 64, 256, 256)
channel_names = ['mem-gfp', 'dpt-dtom']
num_time_points = 1
num_views = 5
num_sheets = 2
big_shape = (2000, 10000, 5000)

convert_czi_views_to_fuse_reg_ZD(path_to_czi_dir, path_to_new_zarr, num_time_points, num_views, num_sheets, namestr, core_file_name, suffix, chunk_sizes, channel_names, big_shape)


