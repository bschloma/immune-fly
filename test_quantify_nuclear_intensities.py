from segmentation import quantify_nuclear_intensities


path_to_zarr = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_06_ca-Gal4_UAS-His-RFP_mNG-EcR-B1/larva_2/im.ome.local_mean.zarr/4'
path_to_labels = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_06_ca-Gal4_UAS-His-RFP_mNG-EcR-B1/larva_2/segmentation.ome.zarr/4'

channel = 0
intensities = quantify_nuclear_intensities(path_to_zarr, path_to_labels, channel=0)