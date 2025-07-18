import zarr
from zarr.storage import DirectoryStore
import numpy as np
import cupy as cp
from tqdm import tqdm
import pandas as pd


path_to_ome_zarr = r''
im = zarr.open(DirectoryStore(r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_06_ca-Gal4_UAS-His-RFP_mNG-EcR-B1/larva_2/im.ome.local_mean.zarr/0'), 'r')
segments = zarr.open(DirectoryStore(r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_06_ca-Gal4_UAS-His-RFP_mNG-EcR-B1/larva_2/segmentation.ome.zarr/0'), 'r')

out_file_name = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_06_ca-Gal4_UAS-His-RFP_mNG-EcR-B1/larva_2/ecr_masked_bkg_sub.zarr'
out_file = zarr.create(store=out_file_name, shape=segments.shape, chunks=im.chunks, dtype=np.float)

# simple masking of the raw gfp channel
# for z in tqdm(range(im.shape[2])):
#     this_im = cp.asarray(im[0, 0, z])
#     this_seg = cp.asarray(segments[0, 0, z])
#     this_masked_im = this_im * (this_seg > 0).astype('uint8')
#     out_file[0, 0, z] = this_masked_im.get()


# color the mask based on background subtracted mean
df = pd.read_pickle(r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_06_ca-Gal4_UAS-His-RFP_mNG-EcR-B1/larva_2/nuclei_3_quant_v4_bkg_by_slice.pkl')
method = 'bkg_sub_mean_ch0'
for z in tqdm(range(im.shape[2])):
    this_im = cp.asarray(im[0, 0, z])
    this_seg = cp.asarray(segments[0, 0, z])
    for nucleus_id in cp.unique(this_seg):
        if nucleus_id == 0:
            continue

        if int(nucleus_id) in df.seg_id.values:
            this_seg[this_seg == nucleus_id] = df[df.seg_id == int(nucleus_id)].get(method).values[0]

    out_file[0, 0, z] = this_seg.get()