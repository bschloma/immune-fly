from readwrite import convert_czi_views_to_fuse_reg_ome_zarr
from pathlib import Path
from glob import glob

experiment_paths = [
    r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2025_04_25_PGRP-LC-GFP_srpHemo-3xmcherry']

czi_paths = []
num_files = []
for path in experiment_paths:
    larvae_dirs = glob(path + '/larva*')
    for larvae_dir in larvae_dirs:
        # if every scan dir is just 'scan_1', use this.
        # czi_paths.append(Path(larvae_dir) / 'scan_1')
        # num_files.append(len(glob(str(Path(larvae_dir) / 'scan_1/*.czi'))))

        # if there are multiple scan dirs per larvae that you want to convert, use this
        scan_dirs = glob(larvae_dir + '/scan_1*')
        for scan_dir in scan_dirs:
            czi_paths.append(Path(scan_dir))
            num_files.append(len(glob(scan_dir + '/*.czi')))


core_file_name = r'scan'
suffix = r'.czi'
namestr = '*.czi'
chunk_sizes = None# (1, 1, 100, 1920, 1920)  # (1, 64, 256, 256)
num_time_points = 1
# num_views = 6
num_sheets = 2
pyramid_scales = 5
reversed_y = False
#sheet_correction = ('green', 'red')
#sheet_correction = {'green': 0, 'red': 1}
sheet_correction = None#{'red': 0}

for i, path in enumerate(czi_paths):
    # if i < 5:
    #     continue
    try:
        path_to_new_zarr = path.parent / 'im.ome.zarr'
        num_views = int(num_files[i] / 2)
        convert_czi_views_to_fuse_reg_ome_zarr(path.__str__(), path_to_new_zarr.__str__(), num_time_points, num_views,
                                               num_sheets, namestr, core_file_name, suffix, chunk_sizes,
                                               pyramid_scales, reversed_y, sheet_correction=sheet_correction)
    except Exception as e:
        print(f'Exception: {e}, skipping!')
        continue


