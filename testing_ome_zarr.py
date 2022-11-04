from readwrite import convert_czi_views_to_fuse_reg_ome_zarr


path_to_czi_dir = r'/media/brandon/Data1/Brandon/fly_immune/Serenity/2022_09_23_dpt-gfp_ecoli-hs-dtom_timeseries/larvae_1/timeseries_1'
path_to_new_zarr = r'/media/brandon/Data1/Brandon/fly_immune/Serenity/2022_09_23_dpt-gfp_ecoli-hs-dtom_timeseries/larvae_1/crop.ome.zarr'
core_file_name = r'timeseries'
suffix = r'.czi'
namestr = '*.czi'
chunk_sizes = (1, 1, 100, 1920, 1920)#(1, 64, 256, 256)
channel_names = ['dpt-gfp', 'bac-dtom']
num_time_points = 18
num_views = 5
num_sheets = 2
big_shape = (1401, num_views * 1920, 9600)
pyramid_scales = 5
reversed_y = False

convert_czi_views_to_fuse_reg_ome_zarr(path_to_czi_dir, path_to_new_zarr, num_time_points, num_views, num_sheets, namestr, core_file_name, suffix, chunk_sizes, channel_names, big_shape, pyramid_scales, reversed_y)



