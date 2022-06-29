""""""


from readwrite import convert_czi_views_to_fuse_reg_ZD

path_to_czi_dir = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_06_17_ca-gal4_uas-mcd8-gfp/larvae_1/zoom_out_scan'
path_to_new_zarr = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_06_17_ca-gal4_uas-mcd8-gfp/larvae_1/zoom_out_scan.zarr'
core_file_name = r'scan'
suffix = r'.czi'
namestr = '*.czi'
chunk_sizes = (1, 100, 1920, 1920)#(1, 64, 256, 256)
channel_names = ['mem-green']
num_time_points = 1
num_views = 4
num_sheets = 2
big_shape = (702, 9600, 5760)

convert_czi_views_to_fuse_reg_ZD(path_to_czi_dir, path_to_new_zarr, num_time_points, num_views, num_sheets, namestr, core_file_name, suffix, chunk_sizes, channel_names, big_shape)


