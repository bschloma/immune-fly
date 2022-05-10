from preprocessing import register_by_stage_coords
import zarr

path_to_fused_zarr = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_1/fused.zarr'
path_to_reg_zarr = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_1/fused.reg.zarr'
first_czi = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_1/full_scan/full_scan.czi'

fused = zarr.open(path_to_fused_zarr, 'r')
reg = register_by_stage_coords(fused, first_czi, path_to_reg_zarr)