from readwrite import run_plantseg_predict

path_to_zarr = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_4/crop.ome.zarr/0'
channel = 1
path_to_pred = r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_4/ds_h5/scan_0/predictions_torch2.zarr'

run_plantseg_predict(path_to_zarr, channel, path_to_pred)