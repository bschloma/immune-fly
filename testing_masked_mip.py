from segmentation import create_mip_mask_by_mem
import napari
import dask.array as da
from skimage.io import imread

path_to_zarr = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2023_05_03-dpt-gfp_r4-gal4_uas-mcd8-mcherry_ecoli-hs-dtom_early-mid_24hrs_high_dose/larvae_2/im.ome.zarr/0'

# params
sigma_blur = 1 #8
beta = 0.5
gamma = 10 ** -0.3
sigma_frangi = 0
thresh = 10 ** -5 #10 ** -1.3
path_to_pred = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2023_05_03-dpt-gfp_r4-gal4_uas-mcd8-mcherry_ecoli-hs-dtom_early-mid_24hrs_high_dose/larvae_2/prediction.zarr'
pred = da.from_zarr(path_to_pred)#[0, 0]
#pred = da.rechunk(pred, chunks=(1, pred.shape[1], pred.shape[2]))

mip_green = create_mip_mask_by_mem(path_to_zarr, sigma_blur, beta, gamma, sigma_frangi, thresh, time_point=0, channel=0, mem=pred)
#mip_red = create_mip_mask_by_mem(path_to_zarr, sigma_blur, beta, gamma, sigma_frangi, thresh, time_point=0, channel=1, mem=pred)

viewer = napari.view_image(mip_green)
#viewer.add_image(mip_red)

#pred_mip = da.max(pred, axis=0).compute()
og_mip_green = imread(r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2023_05_03-dpt-gfp_r4-gal4_uas-mcd8-mcherry_ecoli-hs-dtom_early-mid_24hrs_high_dose/larvae_2/mips/mip_green_0.tif')
og_mip_red = imread(r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2023_05_03-dpt-gfp_r4-gal4_uas-mcd8-mcherry_ecoli-hs-dtom_early-mid_24hrs_high_dose/larvae_2/mips/mip_red_0.tif')

#viewer.add_image(pred_mip)
viewer.add_image(og_mip_green)
viewer.add_image(og_mip_red)