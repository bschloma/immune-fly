from readwrite import convert_czi_views_to_fuse_reg_ome_zarr
from pathlib import Path
from glob import glob

experiment_paths = [
    r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2023_02_07_dpt-gfp_r4-gal4_uas-mcd8-mcherry_ecoli-hs-dtom_earlyL3_24hrs',
    r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2023_02_10_dpt-gfp_4r-gal4_uas-mcd8-mcherry_ecoli-hs-dtom_24hrs',
    r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2023_02_21-dpt-gfp_r4-gal4_uas-mcd8-mcherry_ctl-inj_earlyL3_24hrs',
    r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2023_02_16_dpt-gfp_4r-gal4_uas-mcd8-mcherry_ecoli-hs-dtom_24hrs_mid',
    r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2023_02_24-dpt-gfp_r4-gal4_uas-mcd8-mcherry_noInjCtl_lateL3',
    r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2023_03_01-dpt-gfp_r4-gal4_uas-mcd8-mcherry_ecoli-hs-dtom_earlyL3_24hrs']
czi_paths = []
num_files = []
for path in experiment_paths:
    larvae_dirs = glob(path + '/larvae*')
    for larvae_dir in larvae_dirs:
        czi_paths.append(Path(larvae_dir) / 'scan_1')
        num_files.append(len(glob(str(Path(larvae_dir) / 'scan_1/*.czi'))))

core_file_name = r'scan'
suffix = r'.czi'
namestr = '*.czi'
chunk_sizes = (1, 1, 100, 1920, 1920)  # (1, 64, 256, 256)
num_time_points = 1
# num_views = 6
num_sheets = 2
pyramid_scales = 5
reversed_y = False

for i, path in enumerate(czi_paths):
    try:
        path_to_new_zarr = path.parent / 'im.zarr'
        num_views = int(num_files[i] / 2)
        convert_czi_views_to_fuse_reg_ome_zarr(path.__str__(), path_to_new_zarr.__str__(), num_time_points, num_views,
                                               num_sheets, namestr, core_file_name, suffix, chunk_sizes,
                                               pyramid_scales, reversed_y)
    except Exception(f'error with {path}'):
        continue


########## pasting mip code here temporarily ###############
from readwrite import get_mips
import zarr
import dask.array as da
from glob import glob
from pathlib import Path
from dask.diagnostics import ProgressBar
from PIL import Image


zarr_paths = []
for path in experiment_paths:
    larvae_dirs = glob(path + '/larvae*')
    for larvae_dir in larvae_dirs:
        zarr_paths.append(Path(larvae_dir) / 'tmp.zarr/tmp_big_stack')

for path in zarr_paths:
    if not path.is_dir():
        continue
    try:
        # save mips to tiff using PIL
        mip_dir = Path(path).parent.parent / "mips"
        mip_dir.mkdir(exist_ok=True)
        mip_dir = mip_dir.__str__()

        data = da.from_zarr(path)

        # green
        with ProgressBar():
            green_mip = get_mips(data, channel=0)

        for i in range(green_mip.shape[0]):
            this_green_mip = green_mip[i]
            Image.fromarray(this_green_mip).save(mip_dir + '/mip_green' + '_' + str(i) + '.tif')

        # red
        with ProgressBar():
            red_mip = get_mips(data, channel=1)

        for i in range(green_mip.shape[0]):
            this_red_mip = red_mip[i]
            Image.fromarray(this_red_mip).save(mip_dir + '/mip_red' + '_' + str(i) + '.tif')

    except Exception(f'error with {path}'):
        continue

