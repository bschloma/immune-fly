from readwrite import load_plantseg_model, normalize_std
import zarr
import torch
import dask.array as da
import numpy as np
import napari
import h5py


def process(chunk, model):

    if chunk.shape[0] > 1:
        t = torch.tensor(chunk.astype('float32'), device='cuda')
        t = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(t, dim=0), dim=0), dim=0)
        # call model in test model
        with torch.no_grad():
            t = model(t)

        t = np.array(t.cpu()).squeeze()
    else:
        t = chunk

    return t


zim = zarr.open(r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2023_02_07_dpt-gfp_r4-gal4_uas-mcd8-mcherry_ecoli-hs-dtom_earlyL3_24hrs/larvae_5/tmp.zarr', 'r')
im = zim.tmp_big_stack
model = load_plantseg_model()
model = model.to('cuda')

# pull out a test im
#test_im = im[0, 1, 0, :1024, :1024]

# # convert to tensor
# test_tensor = torch.tensor(test_im.astype('float32'), device='cuda')
#
# # properly expand dims for input to torch model
# torch_tensor = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(test_tensor, dim=0), dim=0), dim=0)
#
# # call model in test model
# with torch.no_grad():
#     out = model(torch_tensor)
f = h5py.File(r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2022_02_24_uas-mcd8-gfp_r4-gal4_x_dipt_dtom2/ecoli_hs_gfp/larvae_4/ds_h5/scan_0/PreProcessing/z_110.h5')
im = f.get('raw')[0]
im = normalize_std(im)
#im_da = da.from_array(im[0, 1, 300], chunks=(256, 256))
im_da = da.from_array(im, chunks=(256, 256))
#result = da.map_overlap(process, im_da, depth=100, model=model).compute()
result = da.map_blocks(process, im_da, model).compute()

#viewer = napari.view_image(result)
