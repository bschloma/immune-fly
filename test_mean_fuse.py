
from preprocessing import simple_mean_fusion
import zarr


path_to_raw_zarr = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_1/full_scan.zarr'
path_to_fused_zarr = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_1/fused.zarr'

raw_zarr = zarr.open(path_to_raw_zarr, 'r')
channel_names = ['488nm', '561nm']

fused = simple_mean_fusion(raw_zarr, path_to_fused_zarr, channel_names)
