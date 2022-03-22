""""""


from readwrite import convert_tiffs_to_zarr


path_to_tiff_dir = r'/media/brandon/Data1/Data/Brandon/fly_larvae/2022_01_06_dpt_dtom_r4_gal4_uas_mcd8_gfp_ecoli_20hrs/larvae_1/scans_culled'
path_to_new_zarr = r'/media/brandon/Data1/Data/Brandon/fly_larvae/2022_01_06_dpt_dtom_r4_gal4_uas_mcd8_gfp_ecoli_20hrs/larvae_1/whole_animal_scan_1.zarr'
chunk_sizes = (64, 256, 256)

root = convert_tiffs_to_zarr(path_to_tiff_dir, path_to_new_zarr, chunk_sizes)


