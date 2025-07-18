from readwrite import convert_czi_views_to_fuse_reg_ome_zarr

path_to_czi_dir = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2023_07_11-mNG-EcR_early_mid/larvae_3/scan_1'
path_to_new_zarr = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2023_07_11-mNG-EcR_early_mid/larvae_3/im.ome.raw_correction.zarr'
core_file_name = r'scan'
suffix = r'.czi'
namestr = '*.czi'
chunk_sizes = None#(1, 1, 100, 800, 1920)#(1, 64, 256, 256)
num_time_points = 1
num_views = 7
num_sheets = 2
pyramid_scales = 5
reversed_y = False
num_channels = 2
sheet_correction = {'green': 0}

convert_czi_views_to_fuse_reg_ome_zarr(path_to_czi_dir, path_to_new_zarr, num_time_points, num_views, num_sheets, namestr, core_file_name, suffix, chunk_sizes, pyramid_scales, sheet_correction=sheet_correction)
