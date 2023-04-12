from readwrite import get_mips
import zarr
import dask.array as da
from glob import glob
from pathlib import Path
from dask.diagnostics import ProgressBar
from PIL import Image


experiment_paths = [
    r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2023_02_16_dpt-gfp_4r-gal4_uas-mcd8-mcherry_ecoli-hs-dtom_24hrs_mid',
    r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2023_02_24-dpt-gfp_r4-gal4_uas-mcd8-mcherry_noInjCtl_lateL3',
    r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2023_03_01-dpt-gfp_r4-gal4_uas-mcd8-mcherry_ecoli-hs-dtom_earlyL3_24hrs']
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

