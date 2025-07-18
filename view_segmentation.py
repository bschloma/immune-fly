import zarr
import napari
import dask.array as da
from zarr.storage import DirectoryStore


path_to_seg = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_20_dpt-gfp_ca-Gal4_UAS-His-RFP_halocarbon_5x_timeseries/larva_2/segmentation.tracked.zarr'
path_to_ome_zarr = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_20_dpt-gfp_ca-Gal4_UAS-His-RFP_halocarbon_5x_timeseries/larva_2/im.ome.zarr'
seg = zarr.open(DirectoryStore(path_to_seg), 'r')
im = da.from_zarr(zarr.open(DirectoryStore(path_to_ome_zarr + '/0')))[:, 1]
viewer = napari.view_image(im)
viewer.add_labels(seg)