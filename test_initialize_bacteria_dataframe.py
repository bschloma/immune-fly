from segmentation import initialize_bacteria_dataframe


path_to_segments = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_03_20_dpt-gfp_r4-gal4_ecoli-hs-dtom_input-output_pilot_4-6hrs/larva_5/bacteria.segmentation_v2.ome.zarr/0'
path_to_im = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_03_20_dpt-gfp_r4-gal4_ecoli-hs-dtom_input-output_pilot_4-6hrs/larva_5/im.ome.no_correction.zarr/0'

dxy = 0.325
dz = 2
channel = 1

df = initialize_bacteria_dataframe(path_to_segments, path_to_im, dxy, dz, channel=channel)