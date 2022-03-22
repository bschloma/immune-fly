""""""


from readwrite import convert_czi_views_to_zarr


path_to_czi_dir = r'/media/brandon/Data1/Data/Brandon/fly_larvae/2022_01_06_dpt_dtom_r4_gal4_uas_mcd8_gfp_ecoli_20hrs/larvae_1/whole_animal_scan_1'
path_to_new_zarr = r'/media/brandon/Data1/Data/Brandon/fly_larvae/2022_01_06_dpt_dtom_r4_gal4_uas_mcd8_gfp_ecoli_20hrs/larvae_1/whole_animal_scan_1.zarr'
core_file_name = r'whole_animal_scan_1'
suffix = r'.czi'
namestr = '*.czi'
chunk_sizes = (1, 64, 256, 256)
channel_names = ['488nm', '561nm']

convert_czi_views_to_zarr(path_to_czi_dir, path_to_new_zarr, namestr, core_file_name, suffix, chunk_sizes, channel_names)


