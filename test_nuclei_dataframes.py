from segmentation import initialize_nuclei_dataframe

path_to_seg = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_06_ca-Gal4_UAS-His-RFP_mNG-EcR-B1/larva_2/segmentation.ome.zarr/3'
path_to_im = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_06_ca-Gal4_UAS-His-RFP_mNG-EcR-B1/larva_2/im.ome.local_mean.zarr/3'


# 5x data
# dxy = 0.9106222
# dz = 4

# 20x data
dxy = 0.325 * (2 ** 3)
dz = 2
channel = 1
voxel_size = (31, 14, 14)

df = initialize_nuclei_dataframe(path_to_seg, dxy=dxy, dz=dz, path_to_im=path_to_im, voxel_size=voxel_size)

#df.to_pickle(r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_20_dpt-gfp_ca-Gal4_UAS-His-RFP_halocarbon_5x_timeseries/larva_2/nuclei.pkl')
