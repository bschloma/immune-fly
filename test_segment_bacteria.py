from segmentation import segment_bacteria


path_to_zarr = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_03_20_dpt-gfp_r4-gal4_ecoli-hs-dtom_input-output_pilot_4-6hrs/larva_4/im.ome.no_correction.zarr/0'
path_to_output_labels_zarr = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_03_20_dpt-gfp_r4-gal4_ecoli-hs-dtom_input-output_pilot_4-6hrs/larva_4/bacteria.segmentation.zarr'

segment_bacteria(path_to_zarr, path_to_output_labels_zarr, sigma_blur=1.0, bright_thresh=0.04, sigma_low=1.0, sigma_high=2.0, disk_size=3, bacteria_thresh=0.0001, time_points=None, maximum_size=None,
                   channel=1, dask_label=True)