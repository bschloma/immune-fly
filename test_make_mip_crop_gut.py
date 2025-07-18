from segmentation import make_mips_crop_gut

path_to_zarr = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2023_05_03-dpt-gfp_r4-gal4_uas-mcd8-mcherry_ecoli-hs-dtom_early-mid_24hrs_high_dose/larvae_1/im.ome.v2.zarr/4'
z_min = 0
z_max = 236
y_min = 0
y_max = 166
channel = 1
path_to_mips = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2023_05_03-dpt-gfp_r4-gal4_uas-mcd8-mcherry_ecoli-hs-dtom_early-mid_24hrs_high_dose/larvae_1/mips4_2_crop_gut'

make_mips_crop_gut(path_to_zarr, path_to_mips=path_to_mips, z_min=z_min, z_max=z_max, y_min=y_min, y_max=y_max, channel=channel)
