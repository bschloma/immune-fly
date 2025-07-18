import napari
import zarr
import dask.array as da


path_to_segments = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_07_02-NP1029-Gal4_ecoli-hs-dtom_2hr/larva_2/bacteria.segmentation.nmax2.zarr'
path_to_im = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_07_02-NP1029-Gal4_ecoli-hs-dtom_2hr/larva_2/im.ome.zarr'


viewer = napari.Viewer()
viewer.open(path_to_im, channel_axis=1, plugin="napari-ome-zarr")
labels = da.from_zarr(zarr.storage.DirectoryStore(path_to_segments))
viewer.add_labels(labels)