import dask.array as da
from zarr.storage import DirectoryStore
from dask_image.ndfilters import gaussian_filter
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster


if __name__ == "__main__":
    cluster = LocalCluster()
    cluster.scale = 2
    client = Client(cluster)

    im = da.from_zarr(DirectoryStore(r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_03_20_dpt-gfp_r4-gal4_ecoli-hs-dtom_input-output_pilot_4-6hrs/larva_2/tmp.density.zarr'))[0, 0]
    im = gaussian_filter(im, sigma=(3, 5, 5))
    im = da.rechunk(im, chunks=(1,) + im.shape[1:])
    out_zarr_store = DirectoryStore(r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_03_20_dpt-gfp_r4-gal4_ecoli-hs-dtom_input-output_pilot_4-6hrs/larva_2/bacteria.density.filt.v2.zarr')
    with ProgressBar():
        im.to_zarr(out_zarr_store)
