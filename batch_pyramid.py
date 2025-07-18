from readwrite import create_pyramid_from_zarr
from glob import glob

experiment_paths = [
    r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_07_02-NP1029-Gal4_ecoli-hs-dtom_2hr']


method = 'nearest'
pyramid_scales = 3
file_name = 'bacteria.segmentation.zarr'
for path in experiment_paths:
    larvae_dirs = glob(path + '/larva*')
    for larvae_dir in larvae_dirs:
        path_to_plain_zarr = larvae_dir + '/' + file_name
        path_to_ome_zarr = larvae_dir + '/bacteria.segmentation.ome.zarr'
        if len(glob(path_to_plain_zarr)) == 0:
            continue
        if len(glob(path_to_ome_zarr)) > 0:
            continue
        print(larvae_dir)
        try:
            create_pyramid_from_zarr(path_to_plain_zarr, path_to_ome_zarr, method=method, pyramid_scales=pyramid_scales)
        except Exception as e:
            print(e)
            continue



