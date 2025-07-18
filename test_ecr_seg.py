from segmentation import segment_ecr, mask_ecr
import zarr


path_to_ome_zarr = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2023_07_11-mNG-EcR_early_mid/larvae_3/im.ome.zarr'
path_to_output_labels = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2023_07_11-mNG-EcR_early_mid/larvae_3/segmentation.zarr'
path_to_masked_zarr = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2023_07_11-mNG-EcR_early_mid/larvae_3/im.masked.zarr'

sigma_low = 8
sigma_high = 24
thresh = 10 ** -3.42
maximum_size = 100_000

#segment_ecr(path_to_ome_zarr, path_to_output_labels, sigma_low, sigma_high, thresh, maximum_size)
mask_ecr(path_to_ome_zarr, path_to_output_labels, path_to_masked_zarr, sigma_low, sigma_high, thresh, maximum_size)
